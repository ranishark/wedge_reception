from clinic import ClinicStore


def test_identify_seeded_patient() -> None:
    store = ClinicStore.seeded()
    result = store.identify_patient("jordan hale", "March 14, 1988")
    assert result["ok"] is True
    assert result["found"] is True
    assert result["patient_id"] == "p-jordan"


def test_identify_unknown_patient() -> None:
    store = ClinicStore.seeded()
    result = store.identify_patient("Alex Nguyen", "2000-01-01")
    assert result["ok"] is True
    assert result["found"] is False


def test_list_seeded_appointments() -> None:
    store = ClinicStore.seeded()
    result = store.list_appointments(name="Jordan Hale", date_of_birth="1988-03-14")
    assert result["ok"] is True
    assert len(result["appointments"]) == 1
    assert result["appointments"][0]["provider"] == "Doctor Maya Chen"
    assert "September" in result["appointments"][0]["when"]


def test_book_open_slot() -> None:
    store = ClinicStore.seeded()
    result = store.book_appointment(
        name="Alex Nguyen",
        date_of_birth="2000-01-01",
        phone="503-555-0100",
        visit_type="new patient",
        provider_name="Doctor Priya Shah",
        date_value="2026-09-22",
        time_value="9:30 AM",
        confirmed=True,
    )
    assert result["ok"] is True
    assert result["appointment"]["visit_type"] == "new patient"
    listed = store.list_appointments(name="Alex Nguyen", date_of_birth="2000-01-01")
    assert len(listed["appointments"]) == 1


def test_book_requires_confirmation() -> None:
    store = ClinicStore.seeded()
    result = store.book_appointment(
        name="Alex Nguyen",
        date_of_birth="2000-01-01",
        phone="503-555-0100",
        visit_type="new patient",
        provider_name="Doctor Priya Shah",
        date_value="2026-09-22",
        time_value="9:30 AM",
        confirmed=False,
    )
    assert result["ok"] is False
    assert result["needs_confirmation"] is True


def test_cancel_seeded_appointment() -> None:
    store = ClinicStore.seeded()
    result = store.cancel_appointment(
        name="Jordan Hale",
        date_of_birth="1988-03-14",
        confirmed=True,
        date_value="2026-09-18",
        time_value="10:00",
    )
    assert result["ok"] is True
    assert result["appointment"]["status"] == "canceled"
    listed = store.list_appointments(name="Jordan Hale", date_of_birth="1988-03-14")
    assert listed["appointments"] == []


def test_reschedule_seeded_appointment() -> None:
    store = ClinicStore.seeded()
    result = store.reschedule_appointment(
        name="Jordan Hale",
        date_of_birth="1988-03-14",
        current_date="2026-09-18",
        current_time="10:00",
        new_date="2026-09-18",
        new_time="11:00",
        confirmed=True,
    )
    assert result["ok"] is True
    assert (
        "11:00" in result["appointment"]["when"]
        or "11:00 AM" in result["appointment"]["when"]
    )
