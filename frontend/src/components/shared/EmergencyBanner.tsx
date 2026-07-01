import { Phone } from 'lucide-react';

export function EmergencyBanner({ message }: { message: string }) {
  return (
    <div className="bubble-emergency" role="alert" aria-live="assertive">
      <div className="flex items-start gap-3">
        <span className="text-2xl select-none flex-shrink-0">🚨</span>
        <div className="flex-1">
          <p className="font-semibold text-red-900 mb-2 leading-snug">{message}</p>
          <a href="tel:112" id="emergency-call-btn" className="inline-flex items-center gap-2 bg-red-600 text-white font-semibold text-sm px-4 py-2 rounded-full hover:bg-red-700 transition-colors">
            <Phone className="w-4 h-4" /> Call 112 Now
          </a>
        </div>
      </div>
      <p className="mt-3 text-xs text-red-700 opacity-80">⚠️ This message cannot be dismissed. Please seek immediate help.</p>
    </div>
  );
}
