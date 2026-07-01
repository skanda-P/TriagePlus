import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'coral' | 'ghost-white' | 'ghost-dark' | 'nav';
  size?: 'sm' | 'md' | 'lg';
}

export function Button({ variant = 'coral', size = 'md', children, className = '', ...props }: ButtonProps) {
  const base = variant === 'coral' ? 'btn-coral' : variant === 'ghost-white' ? 'btn-ghost-white' : variant === 'ghost-dark' ? 'btn-ghost-dark' : 'btn-nav';
  const sizeClass = size === 'sm' ? 'text-sm px-4 py-2' : size === 'lg' ? 'text-lg px-8 py-4' : '';
  return <button className={`${base} ${sizeClass} ${className}`} {...props}>{children}</button>;
}
