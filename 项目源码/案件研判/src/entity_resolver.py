from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import re


EMPTY_VALUES = {None, "", "null", "None", "未知", "-", "nan"}
CARD_PATTERN = re.compile(r"\d{16,19}")


def clean_value(value: Any) -> str | None:
    if value in EMPTY_VALUES:
        return None
    text = str(value).strip()
    if not text or text in EMPTY_VALUES:
        return None
    return text


@dataclass
class EntityDictionary:
    people: dict[str, dict[str, Any]] = field(default_factory=dict)
    account_to_person: dict[str, str] = field(default_factory=dict)
    phone_to_person: dict[str, str] = field(default_factory=dict)

    def add(self, person: str | None, accounts: list[str], phones: list[str]) -> None:
        if not person:
            return
        item = self.people.setdefault(person, {"person": person, "accounts": set(), "phones": set()})
        for account in accounts:
            item["accounts"].add(account)
            self.account_to_person.setdefault(account, person)
        for phone in phones:
            item["phones"].add(phone)
            self.phone_to_person.setdefault(phone, person)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            person: {
                "person": person,
                "accounts": sorted(value["accounts"]),
                "phones": sorted(value["phones"]),
            }
            for person, value in sorted(self.people.items())
        }


def _extract_card_numbers(text: str) -> list[str]:
    return CARD_PATTERN.findall(text)


def _values_from_fields(payload: dict[str, Any], fields: list[str], extract_cards: bool = False) -> list[str]:
    values = []
    for field in fields:
        value = clean_value(payload.get(field))
        if not value:
            continue
        if extract_cards:
            cards = _extract_card_numbers(value)
            if cards:
                values.extend(cards)
                continue
            values.append(value)
        else:
            values.append(value)
    return values


def build_entity_dictionary(rows: list[dict[str, Any]], strategy: dict[str, Any]) -> EntityDictionary:
    dictionary = EntityDictionary()
    person_fields = list(strategy.get("person_fields") or [])
    account_fields = list(strategy.get("account_fields") or [])
    phone_fields = list(strategy.get("phone_fields") or [])
    for payload in rows:
        people = _values_from_fields(payload, person_fields)
        accounts = _values_from_fields(payload, account_fields, extract_cards=True)
        phones = _values_from_fields(payload, phone_fields)
        for person in people:
            dictionary.add(person, accounts, phones)
    return dictionary


def resolve_entity(payload: dict[str, Any], strategy: dict[str, Any], dictionary: EntityDictionary) -> dict[str, Any] | None:
    person_fields = list(strategy.get("person_fields") or [])
    account_fields = list(strategy.get("account_fields") or [])
    phone_fields = list(strategy.get("phone_fields") or [])

    for field in person_fields:
        person = clean_value(payload.get(field))
        if person:
            return {
                "entity_key": f"person:{person}",
                "entity_name": person,
                "entity_type": "person",
                "source_field": field,
                "accounts": _values_from_fields(payload, account_fields, extract_cards=True),
                "phones": _values_from_fields(payload, phone_fields),
            }

    for field in account_fields:
        raw_account = clean_value(payload.get(field))
        if not raw_account:
            continue
        accounts = _extract_card_numbers(raw_account) or [raw_account]
        account = accounts[0]
        person = dictionary.account_to_person.get(account)
        if person:
            return {
                "entity_key": f"person:{person}",
                "entity_name": person,
                "entity_type": "person",
                "source_field": field,
                "accounts": accounts,
                "phones": [],
                "resolved_from": "account_dictionary",
            }
        return {
            "entity_key": f"account:{account}",
            "entity_name": account,
            "entity_type": "account",
            "source_field": field,
            "accounts": accounts,
            "phones": [],
        }

    for field in phone_fields:
        phone = clean_value(payload.get(field))
        if not phone:
            continue
        person = dictionary.phone_to_person.get(phone)
        if person:
            return {
                "entity_key": f"person:{person}",
                "entity_name": person,
                "entity_type": "person",
                "source_field": field,
                "accounts": [],
                "phones": [phone],
                "resolved_from": "phone_dictionary",
            }
        return {
            "entity_key": f"phone:{phone}",
            "entity_name": phone,
            "entity_type": "phone",
            "source_field": field,
            "accounts": [],
            "phones": [phone],
        }
    return None


def merge_sets(target: dict[str, set[str]], accounts: list[str], phones: list[str]) -> None:
    for account in accounts:
        target["accounts"].add(account)
    for phone in phones:
        target["phones"].add(phone)


def collect_entity_mentions(factpacks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for factpack in factpacks:
        context = factpack.get("model_context", {})
        for item in factpack.get("focus_entities", []):
            mentions[item["entity_name"]].append(
                {
                    "model_id": context.get("model_id"),
                    "model_name": context.get("model_name"),
                    "rank": item.get("rank"),
                    "metrics": item.get("metrics", {}),
                    "risk_tags": [flag["risk_tag"] for flag in item.get("anomaly_flags", [])],
                    "fact_id": item.get("fact_id"),
                }
            )
    return mentions
