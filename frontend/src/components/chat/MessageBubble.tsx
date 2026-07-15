import React from 'react';
import { type Message } from '../../stores/chatStore';
import { EmergencyBanner } from '../shared/EmergencyBanner';

function parseMarkdown(text: string): React.ReactNode[] {
  return text.split('\n').map((line, li, arr) => {
    const parts = line.split(/(\*\*[^*]+\*\*)/g).map((part, pi) => {
      if (part.startsWith('**') && part.endsWith('**')) return <strong key={pi}>{part.slice(2,-2)}</strong>;
      return part.split(/(\*[^*]+\*)/g).map((s, si) =>
        s.startsWith('*') && s.endsWith('*') ? <em key={`${pi}-${si}`}>{s.slice(1,-1)}</em> : <span key={`${pi}-${si}`}>{s}</span>
      );
    });
    return <React.Fragment key={li}>{parts}{li < arr.length - 1 && <br />}</React.Fragment>;
  });
}

export function MessageBubble({ message }: { message: Message }) {
  const { role, content, chips } = message;
  
  const handleChipClick = (chip: string) => {
    window.dispatchEvent(new CustomEvent('send-message', { detail: chip }));
  };

  if (role === 'emergency') return <EmergencyBanner message={content} />;
  if (role === 'error') return <div className="flex justify-center my-2"><div className="text-sm font-medium text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2">⚠️ {content}</div></div>;
  if (role === 'patient') return <div className="flex justify-end"><div className="bubble-user">{content}</div></div>;
  
  return (
    <div className="flex flex-col items-start gap-2">
      <div className="bubble-ai prose-chat">{parseMarkdown(content)}</div>
      {chips && chips.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-1">
          {chips.map((chip, i) => (
            <button 
              key={i} 
              onClick={() => handleChipClick(chip)}
              className="px-3 py-1.5 text-sm bg-white dark:bg-slate-800 border border-canopy-green text-canopy-green hover:bg-mint-wash dark:hover:bg-slate-700 rounded-full transition-colors"
            >
              {chip}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
