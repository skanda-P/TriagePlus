import React, { useState } from 'react';
import { Calendar, Clock, User, DollarSign, Mail, Check, X } from 'lucide-react';

interface AppointmentData {
  doctor_name: string;
  specialization: string;
  date: string;
  start_time: string;
  end_time: string;
  consultation_fee: number;
  department: string;
  status: 'pending' | 'confirmed' | 'payment_complete';
}

interface BookingCardProps {
  data: AppointmentData;
  onConfirm: (email: string) => void;
  onCancel: () => void;
}

export function BookingCard({ data, onConfirm, onCancel }: BookingCardProps) {
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [email, setEmail] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleConfirmClick = async () => {
    if (!email || !email.includes('@')) {
      alert('Please enter a valid email');
      return;
    }

    setIsProcessing(true);
    try {
      // Send confirmation
      await fetch('/api/appointments/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          appointment_data: data
        })
      });

      onConfirm(email);
    } catch (error) {
      console.error('Failed to confirm appointment:', error);
      alert('Failed to confirm appointment');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-300 rounded-lg p-6 my-4 space-y-4 max-w-md">
      {/* Header */}
      <div className="flex items-center gap-3 pb-3 border-b border-blue-300">
        <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white">
          <Calendar size={20} />
        </div>
        <div>
          <p className="font-bold text-blue-900">Appointment Confirmed</p>
          <p className="text-sm text-blue-700">Your booking details</p>
        </div>
      </div>

      {/* Appointment Details */}
      <div className="space-y-3">
        {/* Doctor */}
        <div className="flex items-start gap-3">
          <User size={18} className="text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm text-blue-700 font-medium">Doctor</p>
            <p className="font-semibold text-blue-900">{data.doctor_name}</p>
            <p className="text-xs text-blue-600">{data.specialization}</p>
          </div>
        </div>

        {/* Date & Time */}
        <div className="flex items-start gap-3">
          <Calendar size={18} className="text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm text-blue-700 font-medium">Date & Time</p>
            <p className="font-semibold text-blue-900">{formatDate(data.date)}</p>
            <p className="text-sm text-blue-600 flex items-center gap-1 mt-1">
              <Clock size={14} />
              {data.start_time} - {data.end_time}
            </p>
          </div>
        </div>

        {/* Fee */}
        <div className="flex items-start gap-3">
          <DollarSign size={18} className="text-blue-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm text-blue-700 font-medium">Consultation Fee</p>
            <p className="font-semibold text-blue-900">₹{data.consultation_fee.toLocaleString('en-IN')}</p>
          </div>
        </div>
      </div>

      {/* Email Input */}
      {!showEmailInput ? (
        <button
          onClick={() => setShowEmailInput(true)}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          <Mail size={18} />
          Confirm & Send Confirmation Email
        </button>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-blue-900 mb-1">
              Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="your.email@example.com"
              className="w-full px-3 py-2 border border-blue-300 rounded-lg focus:outline-none focus:border-blue-600"
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleConfirmClick}
              disabled={isProcessing}
              className="flex-1 flex items-center justify-center gap-2 bg-green-600 text-white py-2 rounded-lg font-medium hover:bg-green-700 transition-colors disabled:bg-gray-400"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Check size={18} />
                  Confirm
                </>
              )}
            </button>
            <button
              onClick={() => setShowEmailInput(false)}
              className="flex-1 flex items-center justify-center gap-2 bg-gray-400 text-white py-2 rounded-lg font-medium hover:bg-gray-500 transition-colors"
            >
              <X size={18} />
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <p className="text-xs text-blue-700 text-center">
        A confirmation email will be sent to you with all appointment details
      </p>
    </div>
  );
}

export default BookingCard;
