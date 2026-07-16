#!/usr/bin/env python3
"""
Verify synthetic data in TriagePlus database.
Quick check to ensure doctors, patients, and slots are properly populated.
"""

import os
import sys
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend'))

from app.db.supabase_client import get_supabase

def main():
    print("🔍 TriagePlus Data Verification")
    print("=" * 60)
    
    try:
        supabase = get_supabase()
        print("✅ Connected to Supabase\n")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # Check doctors
    print("👨‍⚕️  DOCTORS")
    print("-" * 60)
    try:
        doctors_res = supabase.table("doctor")\
            .select("id, name, rating")\
            .order("created_at", desc=True)\
            .limit(15)\
            .execute()
        
        if doctors_res.data:
            print(f"✅ Found {len(doctors_res.data)} doctors:\n")
            for i, doc in enumerate(doctors_res.data, 1):
                print(f"  {i}. {doc['name']} - ⭐ {doc['rating']}")
        else:
            print("⚠️  No doctors found")
    except Exception as e:
        print(f"❌ Error fetching doctors: {e}")
    
    # Check appointment slots
    print("\n📅 APPOINTMENT SLOTS")
    print("-" * 60)
    try:
        slots_res = supabase.table("clinician_slot")\
            .select("id, start_time, status", count="exact")\
            .eq("status", "open")\
            .execute()
        
        if slots_res.count:
            print(f"✅ Found {slots_res.count} open appointment slots")
            
            # Show distribution by doctor
            doctor_slots = supabase.table("clinician_slot")\
                .select("doctor_id")\
                .eq("status", "open")\
                .execute()
            
            if doctor_slots.data:
                slot_counts = {}
                for slot in doctor_slots.data:
                    doc_id = slot['doctor_id']
                    slot_counts[doc_id] = slot_counts.get(doc_id, 0) + 1
                
                print(f"\n  Slots per doctor:")
                for doc_id, count in sorted(slot_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    - Doctor {doc_id[:8]}...: {count} slots")
        else:
            print("⚠️  No open appointment slots found")
    except Exception as e:
        print(f"❌ Error fetching slots: {e}")
    
    # Check patients
    print("\n👥 PATIENTS")
    print("-" * 60)
    try:
        patients_res = supabase.table("patient")\
            .select("id, name, age, gender, contact")\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        
        if patients_res.data:
            print(f"✅ Found {len(patients_res.data)} patients:\n")
            for i, patient in enumerate(patients_res.data, 1):
                print(f"  {i}. {patient['name']} ({patient['age']}y, {patient['gender']}) - {patient['contact']}")
        else:
            print("⚠️  No patients found")
    except Exception as e:
        print(f"❌ Error fetching patients: {e}")
    
    # Check medical histories
    print("\n🏥 MEDICAL HISTORIES")
    print("-" * 60)
    try:
        history_res = supabase.table("medical_history")\
            .select("id, conditions, medications, allergies", count="exact")\
            .execute()
        
        if history_res.count:
            print(f"✅ Found {history_res.count} medical histories")
            
            # Show sample
            if history_res.data:
                sample = history_res.data[0]
                print(f"\n  Sample history:")
                print(f"    - Conditions: {sample['conditions'] or 'None'}")
                print(f"    - Medications: {sample['medications'] or 'None'}")
                print(f"    - Allergies: {sample['allergies'] or 'None'}")
        else:
            print("⚠️  No medical histories found")
    except Exception as e:
        print(f"❌ Error fetching histories: {e}")
    
    # Summary
    print("\n📊 SUMMARY")
    print("-" * 60)
    try:
        doc_count = supabase.table("doctor").select("*", count="exact").execute().count or 0
        slot_count = supabase.table("clinician_slot").select("*", count="exact").execute().count or 0
        patient_count = supabase.table("patient").select("*", count="exact").execute().count or 0
        history_count = supabase.table("medical_history").select("*", count="exact").execute().count or 0
        
        print(f"  Doctors:           {doc_count:>3}")
        print(f"  Appointment Slots: {slot_count:>3}")
        print(f"  Patients:          {patient_count:>3}")
        print(f"  Medical Histories: {history_count:>3}")
        
        if doc_count > 0 and slot_count > 0 and patient_count > 0:
            print("\n✅ All data populated successfully!")
        else:
            print("\n⚠️  Some data is missing. Run the data generation script.")
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
