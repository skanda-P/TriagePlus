import React, { useState } from 'react';
import { X, CreditCard, Check, AlertCircle } from 'lucide-react';

interface PaymentModalProps {
  appointmentId: string;
  amount: number;
  doctorName: string;
  date: string;
  onSuccess: () => void;
  onCancel: () => void;
}

export function PaymentModal({
  appointmentId,
  amount,
  doctorName,
  date,
  onSuccess,
  onCancel
}: PaymentModalProps) {
  const [step, setStep] = useState<'review' | 'processing' | 'success' | 'error'>('review');
  const [paymentMethod, setPaymentMethod] = useState<'card' | 'upi' | 'net_banking'>('card');
  const [cardDetails, setCardDetails] = useState({
    cardNumber: '',
    expiry: '',
    cvv: ''
  });

  const handlePaymentClick = async () => {
    setStep('processing');
    
    try {
      // Simulate payment processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Create fake payment intent
      const response = await fetch('/api/payments/intent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          appointment_id: appointmentId,
          amount: amount,
          payment_method: paymentMethod
        })
      });

      if (response.ok) {
        setStep('success');
        setTimeout(onSuccess, 2000);
      } else {
        setStep('error');
      }
    } catch (error) {
      console.error('Payment error:', error);
      setStep('error');
    }
  };

  if (step === 'success') {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4 text-center space-y-4">
          <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mx-auto">
            <Check className="w-8 h-8 text-green-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Payment Successful!</h2>
          <p className="text-gray-600">Your appointment has been confirmed.</p>
          <p className="text-sm text-gray-500">Redirecting...</p>
        </div>
      </div>
    );
  }

  if (step === 'error') {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4 text-center space-y-4">
          <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center mx-auto">
            <AlertCircle className="w-8 h-8 text-red-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Payment Failed</h2>
          <p className="text-gray-600">Please try again or use a different payment method.</p>
          <div className="flex gap-2">
            <button
              onClick={() => setStep('review')}
              className="flex-1 bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700"
            >
              Retry
            </button>
            <button
              onClick={onCancel}
              className="flex-1 bg-gray-300 text-gray-900 py-2 rounded-lg font-medium hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg max-w-md w-full mx-4 overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-bold text-white">Complete Payment</h2>
          <button
            onClick={onCancel}
            disabled={step === 'processing'}
            className="text-white hover:text-blue-100"
          >
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Order Summary */}
          <div className="bg-blue-50 rounded-lg p-4 space-y-2">
            <p className="text-sm text-gray-600">Appointment with</p>
            <p className="font-semibold text-gray-900">{doctorName}</p>
            <p className="text-sm text-gray-600">{date}</p>
            <div className="border-t border-blue-200 pt-2 mt-2 flex justify-between">
              <span className="font-medium">Total Amount</span>
              <span className="font-bold text-lg">₹{amount.toLocaleString('en-IN')}</span>
            </div>
          </div>

          {/* Payment Methods */}
          {step === 'review' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-3">
                  Payment Method
                </label>
                <div className="space-y-2">
                  {[
                    { id: 'card', label: 'Credit/Debit Card', icon: CreditCard },
                    { id: 'upi', label: 'UPI', icon: CreditCard },
                    { id: 'net_banking', label: 'Net Banking', icon: CreditCard }
                  ].map(method => (
                    <label
                      key={method.id}
                      className="flex items-center p-3 border-2 rounded-lg cursor-pointer transition-all"
                      style={{
                        borderColor: paymentMethod === method.id ? '#2563eb' : '#e5e7eb',
                        backgroundColor: paymentMethod === method.id ? '#eff6ff' : 'white'
                      }}
                    >
                      <input
                        type="radio"
                        name="payment"
                        value={method.id}
                        checked={paymentMethod === method.id}
                        onChange={e => setPaymentMethod(e.target.value as any)}
                        className="mr-3"
                      />
                      <method.icon size={18} className="mr-2 text-blue-600" />
                      <span className="font-medium">{method.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Card Details (if card selected) */}
              {paymentMethod === 'card' && (
                <div className="space-y-3">
                  <input
                    type="text"
                    placeholder="Card Number"
                    maxLength={19}
                    value={cardDetails.cardNumber}
                    onChange={e => setCardDetails({ ...cardDetails, cardNumber: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600"
                  />
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      type="text"
                      placeholder="MM/YY"
                      maxLength={5}
                      value={cardDetails.expiry}
                      onChange={e => setCardDetails({ ...cardDetails, expiry: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                    <input
                      type="text"
                      placeholder="CVV"
                      maxLength={3}
                      value={cardDetails.cvv}
                      onChange={e => setCardDetails({ ...cardDetails, cvv: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600"
                    />
                  </div>
                </div>
              )}
            </>
          )}

          {/* Processing State */}
          {step === 'processing' && (
            <div className="flex flex-col items-center justify-center py-8">
              <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
              <p className="text-gray-600 font-medium">Processing payment...</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handlePaymentClick}
              disabled={step === 'processing'}
              className="flex-1 bg-green-600 text-white py-3 rounded-lg font-medium hover:bg-green-700 disabled:bg-gray-400 transition-colors flex items-center justify-center gap-2"
            >
              {step === 'processing' ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <CreditCard size={20} />
                  Pay ₹{amount.toLocaleString('en-IN')}
                </>
              )}
            </button>
            <button
              onClick={onCancel}
              disabled={step === 'processing'}
              className="flex-1 bg-gray-300 text-gray-900 py-3 rounded-lg font-medium hover:bg-gray-400 disabled:bg-gray-200 transition-colors"
            >
              Cancel
            </button>
          </div>

          {/* Security Note */}
          <p className="text-xs text-gray-500 text-center">
            🔒 Your payment information is secure and encrypted
          </p>
        </div>
      </div>
    </div>
  );
}

export default PaymentModal;
