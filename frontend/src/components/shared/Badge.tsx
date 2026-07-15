const TRIAGE_LABELS: Record<number, string> = { 1: 'Emergency', 2: 'Urgent', 3: 'Soon', 4: 'Routine' };

interface BadgeProps { label: string; variant?: string; className?: string; }

export function Badge({ label, variant = 'mint', className = '' }: BadgeProps) {
  return <span className={`tag-chip ${variant} ${className}`}>{label}</span>;
}

export function TriageBadge({ level }: { level: number }) {
  const variant = `triage-${Math.max(1, Math.min(4, level))}`;
  const label = TRIAGE_LABELS[level] ?? 'Routine';
  const emoji = level === 1 ? '🔴' : level === 2 ? '🟠' : level === 3 ? '🟡' : '🟢';
  return <span className={`tag-chip ${variant} border`}>{emoji} Level {level} — {label}</span>;
}
