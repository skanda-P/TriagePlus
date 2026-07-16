import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, X, Plus, Copy } from 'lucide-react';

interface TimeSlot {
  start_time: string;
  end_time: string;
  day_of_week: number; // 0-6
}

interface SlotTemplate {
  id: string;
  name: string;
  slots: TimeSlot[];
  active: boolean;
}

interface AvailabilityOverride {
  date: string;
  status: 'available' | 'unavailable' | 'limited';
  reason?: string;
}

export function SlotManagement() {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [selectedDates, setSelectedDates] = useState<Set<string>>(new Set());
  const [templates, setTemplates] = useState<SlotTemplate[]>([]);
  const [overrides, setOverrides] = useState<AvailabilityOverride[]>([]);
  const [showTemplateForm, setShowTemplateForm] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [selectedTimeSlots, setSelectedTimeSlots] = useState<TimeSlot[]>([]);

  // Fetch templates and overrides on mount
  useEffect(() => {
    fetchTemplates();
    fetchOverrides();
  }, []);

  const fetchTemplates = async () => {
    try {
      const response = await fetch('/api/doctor/templates');
      const data = await response.json();
      setTemplates(data);
    } catch (error) {
      console.error('Failed to fetch templates:', error);
    }
  };

  const fetchOverrides = async () => {
    try {
      const response = await fetch('/api/doctor/overrides');
      const data = await response.json();
      setOverrides(data);
    } catch (error) {
      console.error('Failed to fetch overrides:', error);
    }
  };

  const handleDateClick = (date: Date) => {
    const dateStr = date.toISOString().split('T')[0];
    setSelectedDates(prev => {
      const newSet = new Set(prev);
      if (newSet.has(dateStr)) {
        newSet.delete(dateStr);
      } else {
        newSet.add(dateStr);
      }
      return newSet;
    });
  };

  const applyTemplateToSelectedDates = async (template: SlotTemplate) => {
    const dates = Array.from(selectedDates);
    try {
      const response = await fetch('/api/doctor/apply-template', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          template_id: template.id,
          dates: dates,
        }),
      });
      
      if (response.ok) {
        fetchOverrides();
        alert('Template applied successfully');
      }
    } catch (error) {
      console.error('Failed to apply template:', error);
    }
  };

  const createTemplate = async () => {
    if (!templateName || selectedTimeSlots.length === 0) {
      alert('Please enter template name and add time slots');
      return;
    }

    try {
      const response = await fetch('/api/doctor/templates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: templateName,
          slots: selectedTimeSlots,
        }),
      });

      if (response.ok) {
        fetchTemplates();
        setTemplateName('');
        setSelectedTimeSlots([]);
        setShowTemplateForm(false);
      }
    } catch (error) {
      console.error('Failed to create template:', error);
    }
  };

  const cancelSlot = async (date: string, reason: string) => {
    try {
      const response = await fetch('/api/doctor/cancel-slot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date, reason }),
      });

      if (response.ok) {
        fetchOverrides();
        alert('Slot cancelled. Patients with appointments will be notified.');
      }
    } catch (error) {
      console.error('Failed to cancel slot:', error);
    }
  };

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

    // Empty cells for days before month starts
    for (let i = 0; i < firstDay; i++) {
      days.push(<div key={`empty-${i}`} className="h-12"></div>);
    }

    // Days of month
    for (let i = 1; i <= daysInMonth; i++) {
      const date = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), i);
      const dateStr = date.toISOString().split('T')[0];
      const isSelected = selectedDates.has(dateStr);
      const hasOverride = overrides.some(o => o.date === dateStr);

      days.push(
        <button
          key={i}
          onClick={() => handleDateClick(date)}
          className={`h-12 border rounded text-sm font-medium transition-colors ${
            isSelected
              ? 'bg-blue-500 text-white'
              : hasOverride
              ? 'bg-orange-100 text-orange-900'
              : 'hover:bg-gray-100'
          }`}
        >
          {i}
        </button>
      );
    }

    return days;
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4">Slot Management</h2>

        {/* Calendar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Select Dates</h3>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1))}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <ChevronLeft size={20} />
              </button>
              <span className="font-medium w-40 text-center">
                {currentMonth.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
              </span>
              <button
                onClick={() => setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1))}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <ChevronRight size={20} />
              </button>
            </div>
          </div>

          {/* Weekday headers */}
          <div className="grid grid-cols-7 gap-2 mb-2">
            {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
              <div key={day} className="h-10 flex items-center justify-center font-semibold text-gray-600">
                {day}
              </div>
            ))}
          </div>

          {/* Calendar grid */}
          <div className="grid grid-cols-7 gap-2">
            {renderCalendar()}
          </div>

          <p className="text-sm text-gray-600 mt-4">
            Selected: {selectedDates.size} date(s)
          </p>
        </div>

        {/* Templates */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Availability Templates</h3>
            <button
              onClick={() => setShowTemplateForm(!showTemplateForm)}
              className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              <Plus size={20} />
              New Template
            </button>
          </div>

          {showTemplateForm && (
            <div className="bg-blue-50 p-4 rounded mb-4 space-y-4">
              <input
                type="text"
                placeholder="Template name (e.g., 'Mon-Fri 9-5')"
                value={templateName}
                onChange={e => setTemplateName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded"
              />

              <div>
                <label className="block text-sm font-medium mb-2">Time Slots</label>
                <div className="space-y-2">
                  {selectedTimeSlots.map((slot, idx) => (
                    <div key={idx} className="flex items-center gap-2 bg-white p-2 rounded">
                      <span>
                        {slot.start_time} - {slot.end_time}
                      </span>
                      <button
                        onClick={() => setSelectedTimeSlots(selectedTimeSlots.filter((_, i) => i !== idx))}
                        className="ml-auto text-red-500"
                      >
                        <X size={18} />
                      </button>
                    </div>
                  ))}
                  <button
                    onClick={() => setSelectedTimeSlots([...selectedTimeSlots, { start_time: '09:00', end_time: '17:00', day_of_week: 0 }])}
                    className="w-full py-2 border border-dashed border-blue-300 rounded text-blue-600 hover:bg-blue-50"
                  >
                    Add Time Slot
                  </button>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={createTemplate}
                  className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                >
                  Save Template
                </button>
                <button
                  onClick={() => setShowTemplateForm(false)}
                  className="flex-1 bg-gray-300 text-gray-800 px-4 py-2 rounded hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Existing templates */}
          <div className="space-y-2">
            {templates.map(template => (
              <div key={template.id} className="flex items-center justify-between bg-gray-50 p-4 rounded">
                <div>
                  <p className="font-medium">{template.name}</p>
                  <p className="text-sm text-gray-600">{template.slots.length} slots</p>
                </div>
                <button
                  onClick={() => applyTemplateToSelectedDates(template)}
                  className="flex items-center gap-2 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
                  disabled={selectedDates.size === 0}
                >
                  <Copy size={18} />
                  Apply to Selected
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Manage Slots */}
        <div>
          <h3 className="text-lg font-semibold mb-4">Active Slots</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {overrides.map(override => (
              <div key={override.date} className="flex items-center justify-between bg-gray-50 p-4 rounded">
                <div>
                  <p className="font-medium">{new Date(override.date).toLocaleDateString()}</p>
                  <p className="text-sm text-gray-600">{override.status}</p>
                </div>
                <button
                  onClick={() => cancelSlot(override.date, 'Doctor requested cancellation')}
                  className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
                >
                  Cancel
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SlotManagement;
