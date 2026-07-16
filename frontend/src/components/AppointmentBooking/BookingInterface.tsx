import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Search, Clock, MapPin, Star } from 'lucide-react';

interface Doctor {
  id: string;
  name: string;
  specialization: string;
  rating: number;
  available_slots_count: number;
  image_url?: string;
}

interface TimeSlot {
  id: string;
  doctor_id: string;
  date: string;
  start_time: string;
  end_time: string;
  is_booked: boolean;
}

interface BookingStep {
  type: 'department' | 'doctor' | 'symptoms' | 'datetime' | 'payment';
}

export function BookingInterface() {
  const [currentStep, setCurrentStep] = useState<BookingStep['type']>('department');
  const [activeTab, setActiveTab] = useState<'browse' | 'search'>('browse');
  const [selectedDepartment, setSelectedDepartment] = useState<string>('');
  const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Doctor[]>([]);
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [availableSlots, setAvailableSlots] = useState<TimeSlot[]>([]);
  const [selectedSlot, setSelectedSlot] = useState<TimeSlot | null>(null);
  const [symptoms, setSymptoms] = useState('');

  // Departments list
  const departments = [
    'Cardiology',
    'Dermatology',
    'Orthopedics',
    'Neurology',
    'Gastroenterology',
    'General Medicine'
  ];

  // Fetch doctors when department is selected
  useEffect(() => {
    if (selectedDepartment) {
      fetchDoctors();
    }
  }, [selectedDepartment]);

  // Live search
  useEffect(() => {
    if (searchQuery.length > 2) {
      performSearch();
    } else {
      setSearchResults([]);
    }
  }, [searchQuery]);

  const fetchDoctors = async () => {
    try {
      const response = await fetch(`/api/doctors?department=${selectedDepartment}`);
      const data = await response.json();
      setDoctors(data);
    } catch (error) {
      console.error('Failed to fetch doctors:', error);
    }
  };

  const performSearch = async () => {
    try {
      const response = await fetch(`/api/doctors/search?q=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setSearchResults(data);
    } catch (error) {
      console.error('Search failed:', error);
    }
  };

  const selectDoctor = (doctor: Doctor) => {
    setSelectedDoctor(doctor);
    setCurrentStep('symptoms');
  };

  const fetchAvailableSlots = async () => {
    if (!selectedDoctor) return;

    try {
      const response = await fetch(
        `/api/slots?doctor_id=${selectedDoctor.id}&month=${currentMonth.getFullYear()}-${String(currentMonth.getMonth() + 1).padStart(2, '0')}`
      );
      const data = await response.json();
      setAvailableSlots(data);
    } catch (error) {
      console.error('Failed to fetch slots:', error);
    }
  };

  useEffect(() => {
    if (selectedDoctor && currentStep === 'datetime') {
      fetchAvailableSlots();
    }
  }, [currentMonth, selectedDoctor, currentStep]);

  const getDaysInMonth = (date: Date) => {
    return new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
  };

  const getFirstDayOfMonth = (date: Date) => {
    return new Date(date.getFullYear(), date.getMonth(), 1).getDay();
  };

  const renderCalendar = () => {
    const days = [];
    const daysInMonth = getDaysInMonth(currentMonth);
    const firstDay = getFirstDayOfMonth(currentMonth);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Empty cells
    for (let i = 0; i < firstDay; i++) {
      days.push(<div key={`empty-${i}`} className="h-12"></div>);
    }

    // Days
    for (let i = 1; i <= daysInMonth; i++) {
      const date = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), i);
      const dateStr = date.toISOString().split('T')[0];
      const isPast = date < today;
      const daySlotsCount = availableSlots.filter(s => s.date === dateStr && !s.is_booked).length;
      const hasSlots = daySlotsCount > 0;

      days.push(
        <button
          key={i}
          onClick={() => {
            // Clicking a date will show time slots for that day
          }}
          disabled={isPast || !hasSlots}
          className={`h-12 border rounded text-sm font-medium transition-colors relative group ${
            isPast
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : hasSlots
              ? 'bg-white hover:bg-blue-50 border-blue-300 text-blue-600'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          }`}
        >
          {i}
          {hasSlots && (
            <span className="absolute bottom-1 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-blue-500 rounded-full"></span>
          )}
        </button>
      );
    }

    return days;
  };

  // Render time picker for selected date
  const renderTimePicker = () => {
    const selectedDate = new Date(currentMonth);
    const timeSlotsForDate = availableSlots.filter(
      slot => slot.date === selectedDate.toISOString().split('T')[0] && !slot.is_booked
    );

    return (
      <div className="mt-6 space-y-2">
        <h4 className="font-semibold">Available Times</h4>
        <div className="grid grid-cols-3 gap-2">
          {timeSlotsForDate.map(slot => (
            <button
              key={slot.id}
              onClick={() => {
                setSelectedSlot(slot);
                setCurrentStep('payment');
              }}
              className="px-3 py-2 border border-blue-300 rounded hover:bg-blue-50 text-sm font-medium"
            >
              {slot.start_time} - {slot.end_time}
            </button>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      {/* Step indicator */}
      <div className="flex items-center justify-between mb-8">
        {(['department', 'doctor', 'symptoms', 'datetime', 'payment'] as const).map((step, idx) => (
          <div
            key={step}
            className={`flex items-center ${idx < 4 ? 'flex-1' : ''}`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center font-medium text-sm ${
                currentStep === step
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-600'
              }`}
            >
              {idx + 1}
            </div>
            {idx < 4 && (
              <div
                className={`flex-1 h-1 mx-2 ${
                  currentStep === step ? 'bg-blue-600' : 'bg-gray-200'
                }`}
              ></div>
            )}
          </div>
        ))}
      </div>

      {/* Department Selection */}
      {currentStep === 'department' && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Select Department</h2>
          <div className="grid grid-cols-2 gap-3">
            {departments.map(dept => (
              <button
                key={dept}
                onClick={() => {
                  setSelectedDepartment(dept);
                  setCurrentStep('doctor');
                }}
                className="p-4 border-2 border-gray-300 rounded-lg hover:border-blue-600 hover:bg-blue-50 transition-all text-left font-medium"
              >
                {dept}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Doctor Selection */}
      {currentStep === 'doctor' && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Select Doctor</h2>

          {/* Tabs */}
          <div className="flex gap-4 border-b">
            <button
              onClick={() => setActiveTab('browse')}
              className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                activeTab === 'browse'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              Browse by Department
            </button>
            <button
              onClick={() => setActiveTab('search')}
              className={`px-4 py-2 font-medium border-b-2 transition-colors ${
                activeTab === 'search'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-900'
              }`}
            >
              Search Doctor
            </button>
          </div>

          {/* Browse Tab */}
          {activeTab === 'browse' && (
            <div className="space-y-3">
              {doctors.map(doctor => (
                <div
                  key={doctor.id}
                  onClick={() => selectDoctor(doctor)}
                  className="p-4 border border-gray-300 rounded-lg hover:border-blue-600 hover:shadow-lg cursor-pointer transition-all"
                >
                  <div className="flex items-start gap-4">
                    {doctor.image_url && (
                      <img
                        src={doctor.image_url}
                        alt={doctor.name}
                        className="w-16 h-16 rounded-full object-cover"
                      />
                    )}
                    <div className="flex-1">
                      <h3 className="font-semibold text-lg">{doctor.name}</h3>
                      <p className="text-gray-600">{doctor.specialization}</p>
                      <div className="flex items-center gap-4 mt-2 text-sm">
                        <div className="flex items-center gap-1">
                          <Star size={16} className="text-yellow-400" fill="currentColor" />
                          <span>{doctor.rating.toFixed(1)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock size={16} />
                          <span>{doctor.available_slots_count} slots available</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Search Tab */}
          {activeTab === 'search' && (
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-3 text-gray-400" size={20} />
                <input
                  type="text"
                  placeholder="Search doctor by name..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600"
                />
              </div>

              <div className="space-y-3">
                {searchResults.map(doctor => (
                  <div
                    key={doctor.id}
                    onClick={() => selectDoctor(doctor)}
                    className="p-4 border border-gray-300 rounded-lg hover:border-blue-600 hover:shadow-lg cursor-pointer transition-all"
                  >
                    <div className="flex items-start gap-4">
                      {doctor.image_url && (
                        <img
                          src={doctor.image_url}
                          alt={doctor.name}
                          className="w-16 h-16 rounded-full object-cover"
                        />
                      )}
                      <div className="flex-1">
                        <h3 className="font-semibold text-lg">{doctor.name}</h3>
                        <p className="text-gray-600">{doctor.specialization}</p>
                        <div className="flex items-center gap-4 mt-2 text-sm">
                          <div className="flex items-center gap-1">
                            <Star size={16} className="text-yellow-400" fill="currentColor" />
                            <span>{doctor.rating.toFixed(1)}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock size={16} />
                            <span>{doctor.available_slots_count} slots</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Symptoms */}
      {currentStep === 'symptoms' && selectedDoctor && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Describe Your Symptoms (Optional)</h2>
          <div className="space-y-3">
            <textarea
              value={symptoms}
              onChange={e => setSymptoms(e.target.value)}
              placeholder="Tell us about your symptoms..."
              className="w-full p-4 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600 resize-none h-32"
            />
            <button
              onClick={() => setCurrentStep('datetime')}
              className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700"
            >
              Continue to Select Date & Time
            </button>
          </div>
        </div>
      )}

      {/* Date & Time Selection */}
      {currentStep === 'datetime' && selectedDoctor && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Select Date & Time</h2>
          <p className="text-gray-600">Dr. {selectedDoctor.name}</p>

          {/* Calendar */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1))}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <ChevronLeft size={20} />
              </button>
              <span className="font-medium">
                {currentMonth.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
              </span>
              <button
                onClick={() => setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1))}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <ChevronRight size={20} />
              </button>
            </div>

            {/* Weekday headers */}
            <div className="grid grid-cols-7 gap-2">
              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                <div key={day} className="h-10 flex items-center justify-center text-xs font-semibold text-gray-600">
                  {day}
                </div>
              ))}
            </div>

            {/* Calendar grid */}
            <div className="grid grid-cols-7 gap-2">
              {renderCalendar()}
            </div>

            {/* Time picker */}
            {renderTimePicker()}
          </div>
        </div>
      )}

      {/* Payment */}
      {currentStep === 'payment' && selectedSlot && selectedDoctor && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">Confirm & Pay</h2>

          {/* Appointment Summary */}
          <div className="bg-blue-50 p-6 rounded-lg space-y-3">
            <h3 className="font-semibold text-lg">Appointment Summary</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Doctor:</span>
                <span className="font-medium">{selectedDoctor.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Specialization:</span>
                <span className="font-medium">{selectedDoctor.specialization}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Date:</span>
                <span className="font-medium">{new Date(selectedSlot.date).toLocaleDateString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Time:</span>
                <span className="font-medium">{selectedSlot.start_time} - {selectedSlot.end_time}</span>
              </div>
              <div className="border-t pt-2 flex justify-between">
                <span className="font-semibold">Consultation Fee:</span>
                <span className="font-bold">₹1,500</span>
              </div>
            </div>
          </div>

          {/* Payment Button */}
          <button className="w-full bg-green-600 text-white py-3 rounded-lg font-medium hover:bg-green-700">
            Proceed to Payment
          </button>

          <button
            onClick={() => setCurrentStep('datetime')}
            className="w-full border border-gray-300 py-3 rounded-lg font-medium hover:bg-gray-50"
          >
            Back
          </button>
        </div>
      )}
    </div>
  );
}

export default BookingInterface;
