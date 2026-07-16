#!/usr/bin/env python3
"""
Generate synthetic doctor and appointment slot data for TriagePlus.
This script creates realistic doctor profiles with availability slots.
"""

import os
import sys
from datetime import datetime, timedelta
import random
from typing import List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend'))

from app.db.supabase_client import get_supabase

# Doctor names and specialties
INDIAN_DOCTOR_NAMES = [
    "Dr. Rajesh Kumar", "Dr. Priya Sharma", "Dr. Amit Patel", "Dr. Neha Singh",
    "Dr. Vikram Gupta", "Dr. Anjali Verma", "Dr. Sanjay Reddy", "Dr. Pooja Desai",
    "Dr. Arjun Nair", "Dr. Sneha Menon", "Dr. Rohan Sharma", "Dr. Divya Iyer",
    "Dr. Arun Pillai", "Dr. Ravi Chandran", "Dr. Meera Chopra", "Dr. Akshay Singh",
    "Dr. Deepika Roy", "Dr. Varun Kapoor", "Dr. Shruti Malhotra", "Dr. Nikhil Joshi",
    "Dr. Shreya Gupta", "Dr. Aryan Mishra", "Dr. Isha Bhat", "Dr. Rohit Tiwari",
    "Dr. Aarav Kulkarni", "Dr. Nisha Saxena", "Dr. Mohit Bhatnagar", "Dr. Priya Nambiar",
    "Dr. Siddharth Garg", "Dr. Aisha Khan", "Dr. Kabir Ahmad", "Dr. Zara Ali"
]

SPECIALTIES = [
    "Cardiology", "Dermatology", "Orthopedics", "Gastroenterology",
    "Neurology", "Pediatrics", "Psychiatry", "Respiratory",
    "General Medicine / Internal Medicine"
]

def create_synthetic_doctors(num_doctors: int = 15) -> List[dict]:
    """Generate synthetic doctor data"""
    doctors = []
    used_names = set()
    
    for _ in range(num_doctors):
        # Pick unique name
        name = random.choice(INDIAN_DOCTOR_NAMES)
        while name in used_names:
            name = random.choice(INDIAN_DOCTOR_NAMES)
        used_names.add(name)
        
        # Random specialty
        specialty = random.choice(SPECIALTIES)
        
        # Random rating between 4.0 and 5.0
        rating = round(random.uniform(4.0, 5.0), 1)
        
        # Random consultation duration 15-45 minutes
        avg_consult_min = random.choice([15, 20, 30, 45])
        
        doctor = {
            "name": name,
            "specialty": specialty,
            "rating": rating,
            "avg_consult_min": avg_consult_min,
        }
        doctors.append(doctor)
    
    return doctors

def insert_doctors_to_db(doctors: List[dict]):
    """Insert doctors into Supabase"""
    supabase = get_supabase()
    
    print(f"\n📋 Inserting {len(doctors)} doctors...")
    
    for doctor in doctors:
        try:
            # Get specialty ID
            specialty_res = supabase.table("specialty")\
                .select("id")\
                .eq("name", doctor["specialty"])\
                .execute()
            
            if not specialty_res.data:
                print(f"⚠️  Specialty '{doctor['specialty']}' not found, skipping doctor {doctor['name']}")
                continue
            
            specialty_id = specialty_res.data[0]["id"]
            
            # Insert doctor
            doctor_data = {
                "name": doctor["name"],
                "specialty_id": specialty_id,
                "rating": doctor["rating"],
                "avg_consult_min": doctor["avg_consult_min"],
            }
            
            doctor_res = supabase.table("doctor").insert(doctor_data).execute()
            
            if doctor_res.data:
                doctor_id = doctor_res.data[0]["id"]
                print(f"✅ Created doctor: {doctor['name']} ({doctor['specialty']}) - ID: {doctor_id}")
                
                # Create appointment slots for this doctor
                create_slots_for_doctor(supabase, doctor_id, doctor["name"])
            else:
                print(f"❌ Failed to create doctor: {doctor['name']}")
                
        except Exception as e:
            print(f"❌ Error inserting doctor {doctor['name']}: {e}")

def create_slots_for_doctor(supabase, doctor_id: str, doctor_name: str, num_slots: int = 20):
    """Create appointment slots for a doctor"""
    print(f"   Creating {num_slots} appointment slots for {doctor_name}...")
    
    base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    
    for day_offset in range(1, 5):  # Next 4 days
        for hour in [9, 10, 11, 14, 15, 16, 17]:  # 9-12am, 2-5pm
            for minute in [0, 30]:
                slot_time = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)
                
                # Skip if it's a weekend
                if slot_time.weekday() >= 5:
                    continue
                
                try:
                    slot_data = {
                        "doctor_id": doctor_id,
                        "start_time": slot_time.isoformat(),
                        "status": "open"
                    }
                    
                    slot_res = supabase.table("clinician_slot").insert(slot_data).execute()
                    
                    if not slot_res.data:
                        print(f"   ⚠️  Failed to create slot at {slot_time}")
                        
                except Exception as e:
                    print(f"   ❌ Error creating slot: {e}")
    
    print(f"   ✅ Created slots for {doctor_name}")

FIRST_NAMES = [
    "Raj", "Priya", "Amit", "Neha", "Vikram", "Anjali", "Sanjay", "Pooja",
    "Arjun", "Sneha", "Rohan", "Divya", "Arun", "Ravi", "Meera", "Akshay",
    "Deepika", "Varun", "Shruti", "Nikhil", "Shreya", "Aryan", "Isha", "Rohit",
    "Aarav", "Nisha", "Mohit", "Siddharth", "Aisha", "Kabir", "Zara", "Anuj",
    "Harsh", "Karan", "Manoj", "Naveen", "Pankaj", "Sachin", "Tarun", "Uday",
    "Vikash", "Wasim", "Xavier", "Yogesh", "Zainab", "Aditi", "Bhavna", "Chanda"
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Gupta", "Verma", "Reddy", "Desai", "Nair",
    "Menon", "Iyer", "Pillai", "Chandran", "Chopra", "Roy", "Kapoor", "Malhotra",
    "Joshi", "Bhat", "Tiwari", "Kulkarni", "Saxena", "Bhatnagar", "Nambiar", "Garg",
    "Khan", "Ahmad", "Ali", "Kumar", "Rao", "Bansal", "Vyas", "Das",
    "Mishra", "Bose", "Dutta", "Sinha", "Agarwal", "Arora", "Pandey", "Yadav"
]

def create_synthetic_patients(num_patients: int = 10) -> List[dict]:
    """Generate synthetic patient data"""
    patients = []
    
    for _ in range(num_patients):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        patient = {
            "name": f"{first_name} {last_name}",
            "age": random.randint(5, 85),
            "gender": random.choice(["Male", "Female", "Other"]),
            "contact": f"+91{random.randint(6000000000, 9999999999)}",
            "language": random.choice(["en", "hi", "ta", "te"]),
        }
        patients.append(patient)
    
    return patients

def insert_patients_to_db(patients: List[dict]):
    """Insert patients into Supabase"""
    supabase = get_supabase()
    
    print(f"\n👥 Inserting {len(patients)} patients...")
    
    for patient in patients:
        try:
            patient_res = supabase.table("patient").insert(patient).execute()
            
            if patient_res.data:
                patient_id = patient_res.data[0]["id"]
                print(f"✅ Created patient: {patient['name']} ({patient['age']} years) - ID: {patient_id}")
                
                # Create medical history
                create_medical_history(supabase, patient_id, patient["name"])
            else:
                print(f"❌ Failed to create patient: {patient['name']}")
                
        except Exception as e:
            print(f"❌ Error inserting patient {patient['name']}: {e}")

def create_medical_history(supabase, patient_id: str, patient_name: str):
    """Create medical history for a patient"""
    conditions = random.choice([
        "Diabetes, Hypertension",
        "Asthma",
        "Migraine",
        "Arthritis",
        None
    ])
    
    medications = random.choice([
        "Metformin, Lisinopril",
        "Albuterol inhaler",
        "Aspirin",
        None
    ])
    
    allergies = random.choice([
        "Penicillin",
        "Latex",
        "Shellfish",
        None
    ])
    
    history_data = {
        "patient_id": patient_id,
        "conditions": conditions,
        "medications": medications,
        "allergies": allergies,
        "immunocompromised": random.choice([True, False])
    }
    
    try:
        supabase.table("medical_history").insert(history_data).execute()
        print(f"   ✅ Created medical history for {patient_name}")
    except Exception as e:
        print(f"   ❌ Error creating medical history: {e}")

def verify_data():
    """Verify the data was inserted"""
    supabase = get_supabase()
    
    print("\n📊 Data Verification:")
    
    doctors_res = supabase.table("doctor").select("*", count="exact").execute()
    print(f"   Doctors in database: {doctors_res.count if doctors_res else 0}")
    
    slots_res = supabase.table("clinician_slot").select("*", count="exact").execute()
    print(f"   Appointment slots in database: {slots_res.count if slots_res else 0}")
    
    patients_res = supabase.table("patient").select("*", count="exact").execute()
    print(f"   Patients in database: {patients_res.count if patients_res else 0}")
    
    history_res = supabase.table("medical_history").select("*", count="exact").execute()
    print(f"   Medical histories in database: {history_res.count if history_res else 0}")

def main():
    print("🚀 TriagePlus Synthetic Data Generator")
    print("=" * 50)
    
    # Verify Supabase connection
    try:
        supabase = get_supabase()
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # Generate and insert doctors
    print("\n📝 Generating synthetic doctors...")
    doctors = create_synthetic_doctors(num_doctors=15)
    insert_doctors_to_db(doctors)
    
    # Generate and insert patients
    print("\n📝 Generating synthetic patients...")
    patients = create_synthetic_patients(num_patients=10)
    insert_patients_to_db(patients)
    
    # Verify data
    verify_data()
    
    print("\n✅ Data generation complete!")
    print("   You can now log in as a doctor or test the triage system")

if __name__ == "__main__":
    main()
