from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any

CLINIC_NAME = "Wedge Family Practice"
CLINIC_INFO: dict[str, Any] = {
    "name": CLINIC_NAME,
    "address": "214 Maple Street, Suite 100, Portland, Oregon",
    "phone": "503-555-0140",
    "hours": (
        "Monday through Friday, eight in the morning until five in the evening. "
        "Closed on weekends and holidays."
    ),
    "parking": "Free parking in the lot behind the building. Enter from Oak Avenue.",
    "insurance": (
        "We accept Blue Cross Blue Shield, Aetna, Medicare, and most P P O plans. "
        "For other plans, the front desk can check coverage."
    ),
    "visit_types": (
        "annual checkup",
        "follow-up",
        "new patient",
        "sick visit",
    ),
    "visit_prep": (
        "Please arrive fifteen minutes early with a photo I D and insurance card. "
        "Bring a list of current medications. The clinic does not give medical advice "
        "over the phone."
    ),
}

VISIT_TYPES = CLINIC_INFO["visit_types"]


def _norm_name(value: str) -> str:
    return " ".join(value.casefold().split())


def parse_date(value: str) -> date | None:
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_time(value: str) -> time | None:
    text = value.strip().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
        try:
            return datetime.strptime(text, fmt).time().replace(second=0, microsecond=0)
        except ValueError:
            continue
    return None


def _speak_when(day: date, slot: time) -> str:
    dt = datetime.combine(day, slot)
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{dt.strftime('%A, %B')} {day.day}, {day.year} at {hour}:{dt.strftime('%M %p')}"


@dataclass
class Patient:
    patient_id: str
    name: str
    date_of_birth: date
    phone: str


@dataclass
class Provider:
    provider_id: str
    name: str
    specialty: str


@dataclass
class Appointment:
    reference: str
    patient_id: str
    provider_id: str
    visit_type: str
    day: date
    slot: time
    status: str = "scheduled"


@dataclass
class ClinicStore:
    patients: dict[str, Patient] = field(default_factory=dict)
    providers: dict[str, Provider] = field(default_factory=dict)
    appointments: dict[str, Appointment] = field(default_factory=dict)
    open_slots: list[tuple[str, date, time]] = field(default_factory=list)
    _next_reference: int = 1003
    _next_patient: int = 3

    @classmethod
    def seeded(cls) -> ClinicStore:
        store = cls()
        store.providers = {
            "chen": Provider("chen", "Doctor Maya Chen", "family medicine"),
            "ortega": Provider("ortega", "Doctor Luis Ortega", "family medicine"),
            "shah": Provider("shah", "Doctor Priya Shah", "family medicine"),
        }
        store.patients = {
            "p-jordan": Patient(
                "p-jordan", "Jordan Hale", date(1988, 3, 14), "503-555-0199"
            ),
            "p-sam": Patient("p-sam", "Sam Rivera", date(1992, 11, 2), "503-555-0177"),
        }
        store.appointments = {
            "A-1001": Appointment(
                "A-1001",
                "p-jordan",
                "chen",
                "follow-up",
                date(2026, 9, 18),
                time(10, 0),
            ),
            "A-1002": Appointment(
                "A-1002",
                "p-sam",
                "ortega",
                "annual checkup",
                date(2026, 9, 21),
                time(14, 0),
            ),
        }
        store.open_slots = [
            ("chen", date(2026, 9, 18), time(9, 0)),
            ("chen", date(2026, 9, 18), time(11, 0)),
            ("chen", date(2026, 9, 22), time(14, 0)),
            ("ortega", date(2026, 9, 21), time(9, 0)),
            ("ortega", date(2026, 9, 22), time(10, 0)),
            ("shah", date(2026, 9, 22), time(9, 30)),
            ("shah", date(2026, 9, 23), time(15, 0)),
        ]
        return store

    def clinic_info(self) -> dict[str, Any]:
        return dict(CLINIC_INFO)

    def identify_patient(self, name: str, date_of_birth: str) -> dict[str, Any]:
        dob = parse_date(date_of_birth)
        if dob is None:
            return {
                "ok": False,
                "error": "I could not read that date of birth. Please use month, day, and year.",
            }
        wanted = _norm_name(name)
        for patient in self.patients.values():
            if _norm_name(patient.name) == wanted and patient.date_of_birth == dob:
                return {
                    "ok": True,
                    "found": True,
                    "name": patient.name,
                    "patient_id": patient.patient_id,
                }
        return {
            "ok": True,
            "found": False,
            "message": "No matching patient is on file. A new patient visit can still be booked.",
        }

    def _resolve_patient(
        self,
        *,
        patient_id: str | None = None,
        name: str | None = None,
        date_of_birth: str | None = None,
        phone: str | None = None,
        create_if_missing: bool = False,
    ) -> tuple[Patient | None, dict[str, Any] | None]:
        if patient_id and patient_id in self.patients:
            return self.patients[patient_id], None
        if name and date_of_birth:
            identified = self.identify_patient(name, date_of_birth)
            if identified.get("found"):
                return self.patients[identified["patient_id"]], None
            if create_if_missing:
                if not phone:
                    return None, {
                        "ok": False,
                        "error": "A callback phone number is required to add a new patient.",
                    }
                dob = parse_date(date_of_birth)
                if dob is None:
                    return None, {
                        "ok": False,
                        "error": "I could not read that date of birth.",
                    }
                new_id = f"p-{self._next_patient}"
                self._next_patient += 1
                patient = Patient(new_id, name.strip(), dob, phone.strip())
                self.patients[new_id] = patient
                return patient, None
            return None, {
                "ok": False,
                "error": "No matching patient is on file. Collect name, date of birth, and phone to add them.",
            }
        return None, {
            "ok": False,
            "error": "Identify the patient with full name and date of birth first.",
        }

    def _find_provider(self, provider_name: str | None) -> Provider | None:
        if not provider_name:
            return None
        needle = _norm_name(provider_name).replace("doctor ", "").replace("dr ", "")
        for provider in self.providers.values():
            if needle in _norm_name(provider.name) or needle == provider.provider_id:
                return provider
        return None

    def _appointment_view(self, appt: Appointment) -> dict[str, Any]:
        patient = self.patients[appt.patient_id]
        provider = self.providers[appt.provider_id]
        return {
            "reference": appt.reference,
            "patient_name": patient.name,
            "provider": provider.name,
            "visit_type": appt.visit_type,
            "when": _speak_when(appt.day, appt.slot),
            "status": appt.status,
        }

    def list_appointments(
        self,
        *,
        patient_id: str | None = None,
        name: str | None = None,
        date_of_birth: str | None = None,
    ) -> dict[str, Any]:
        patient, err = self._resolve_patient(
            patient_id=patient_id, name=name, date_of_birth=date_of_birth
        )
        if err:
            return err
        assert patient is not None
        items = [
            self._appointment_view(appt)
            for appt in self.appointments.values()
            if appt.patient_id == patient.patient_id and appt.status == "scheduled"
        ]
        return {"ok": True, "appointments": items}

    def list_availability(
        self,
        *,
        visit_type: str | None = None,
        provider_name: str | None = None,
        preferred_date: str | None = None,
    ) -> dict[str, Any]:
        if visit_type and visit_type.casefold() not in {
            v.casefold() for v in VISIT_TYPES
        }:
            return {
                "ok": False,
                "error": "That visit type is not offered. Choose annual checkup, follow-up, new patient, or sick visit.",
            }
        day = parse_date(preferred_date) if preferred_date else None
        if preferred_date and day is None:
            return {"ok": False, "error": "I could not read that preferred date."}
        provider = self._find_provider(provider_name) if provider_name else None
        if provider_name and provider is None:
            return {"ok": False, "error": "That provider is not at this clinic."}

        options: list[dict[str, Any]] = []
        for provider_id, slot_day, slot_time in self.open_slots:
            if provider and provider_id != provider.provider_id:
                continue
            if day and slot_day != day:
                continue
            options.append(
                {
                    "provider": self.providers[provider_id].name,
                    "when": _speak_when(slot_day, slot_time),
                    "date": slot_day.isoformat(),
                    "time": slot_time.strftime("%H:%M"),
                }
            )
            if len(options) >= 5:
                break
        return {"ok": True, "open_times": options}

    def _take_slot(self, provider_id: str, day: date, slot: time) -> bool:
        match = (provider_id, day, slot)
        if match not in self.open_slots:
            return False
        self.open_slots.remove(match)
        return True

    def _release_slot(self, provider_id: str, day: date, slot: time) -> None:
        key = (provider_id, day, slot)
        if key not in self.open_slots:
            self.open_slots.append(key)

    def book_appointment(
        self,
        *,
        name: str,
        date_of_birth: str,
        phone: str,
        visit_type: str,
        provider_name: str,
        date_value: str,
        time_value: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        if not confirmed:
            return {
                "ok": False,
                "needs_confirmation": True,
                "error": "Confirm the visit details with the caller before booking.",
            }
        if visit_type.casefold() not in {v.casefold() for v in VISIT_TYPES}:
            return {"ok": False, "error": "That visit type is not offered."}
        canonical_type = next(
            v for v in VISIT_TYPES if v.casefold() == visit_type.casefold()
        )
        day = parse_date(date_value)
        slot = parse_time(time_value)
        if day is None or slot is None:
            return {"ok": False, "error": "I could not read that date or time."}
        provider = self._find_provider(provider_name)
        if provider is None:
            return {"ok": False, "error": "That provider is not at this clinic."}
        if not self._take_slot(provider.provider_id, day, slot):
            return {"ok": False, "error": "That time is no longer available."}
        patient, err = self._resolve_patient(
            name=name,
            date_of_birth=date_of_birth,
            phone=phone,
            create_if_missing=True,
        )
        if err:
            self._release_slot(provider.provider_id, day, slot)
            return err
        assert patient is not None
        reference = f"A-{self._next_reference}"
        self._next_reference += 1
        appt = Appointment(
            reference,
            patient.patient_id,
            provider.provider_id,
            canonical_type,
            day,
            slot,
        )
        self.appointments[reference] = appt
        return {"ok": True, "appointment": self._appointment_view(appt)}

    def _find_scheduled(
        self,
        patient: Patient,
        *,
        reference: str | None = None,
        date_value: str | None = None,
        time_value: str | None = None,
    ) -> tuple[Appointment | None, dict[str, Any] | None]:
        if reference:
            appt = self.appointments.get(reference.strip())
            if (
                appt is None
                or appt.patient_id != patient.patient_id
                or appt.status != "scheduled"
            ):
                return None, {
                    "ok": False,
                    "error": "That appointment could not be found.",
                }
            return appt, None
        day = parse_date(date_value) if date_value else None
        slot = parse_time(time_value) if time_value else None
        matches = [
            appt
            for appt in self.appointments.values()
            if appt.patient_id == patient.patient_id
            and appt.status == "scheduled"
            and (day is None or appt.day == day)
            and (slot is None or appt.slot == slot)
        ]
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, {
                "ok": False,
                "error": "No matching scheduled appointment was found.",
            }
        return None, {
            "ok": False,
            "error": "More than one appointment matched. Ask which date and time to change.",
            "appointments": [self._appointment_view(item) for item in matches],
        }

    def cancel_appointment(
        self,
        *,
        name: str,
        date_of_birth: str,
        confirmed: bool,
        reference: str | None = None,
        date_value: str | None = None,
        time_value: str | None = None,
    ) -> dict[str, Any]:
        if not confirmed:
            return {
                "ok": False,
                "needs_confirmation": True,
                "error": "Confirm with the caller before canceling.",
            }
        patient, err = self._resolve_patient(name=name, date_of_birth=date_of_birth)
        if err:
            return err
        assert patient is not None
        appt, find_err = self._find_scheduled(
            patient, reference=reference, date_value=date_value, time_value=time_value
        )
        if find_err:
            return find_err
        assert appt is not None
        appt.status = "canceled"
        self._release_slot(appt.provider_id, appt.day, appt.slot)
        return {"ok": True, "appointment": self._appointment_view(appt)}

    def reschedule_appointment(
        self,
        *,
        name: str,
        date_of_birth: str,
        new_date: str,
        new_time: str,
        confirmed: bool,
        reference: str | None = None,
        provider_name: str | None = None,
        current_date: str | None = None,
        current_time: str | None = None,
    ) -> dict[str, Any]:
        if not confirmed:
            return {
                "ok": False,
                "needs_confirmation": True,
                "error": "Confirm the new date and time with the caller before rescheduling.",
            }
        patient, err = self._resolve_patient(name=name, date_of_birth=date_of_birth)
        if err:
            return err
        assert patient is not None
        appt, find_err = self._find_scheduled(
            patient,
            reference=reference,
            date_value=current_date,
            time_value=current_time,
        )
        if find_err:
            return find_err
        assert appt is not None
        day = parse_date(new_date)
        slot = parse_time(new_time)
        if day is None or slot is None:
            return {"ok": False, "error": "I could not read the new date or time."}
        provider_id = appt.provider_id
        if provider_name:
            provider = self._find_provider(provider_name)
            if provider is None:
                return {"ok": False, "error": "That provider is not at this clinic."}
            provider_id = provider.provider_id
        if not self._take_slot(provider_id, day, slot):
            return {"ok": False, "error": "That new time is no longer available."}
        self._release_slot(appt.provider_id, appt.day, appt.slot)
        appt.provider_id = provider_id
        appt.day = day
        appt.slot = slot
        return {"ok": True, "appointment": self._appointment_view(appt)}


@dataclass
class ClinicSession:
    store: ClinicStore
    identified_patient_id: str | None = None
    greeted: bool = False
