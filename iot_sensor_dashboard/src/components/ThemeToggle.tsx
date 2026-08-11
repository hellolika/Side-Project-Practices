import React from 'react';
import { Button } from '@/components/ui/button';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={toggleTheme}
      className={`gap-2 transition-colors duration-200 ${
        theme === 'light'
          ? 'bg-slate-100 border-slate-300 hover:bg-slate-200 text-slate-700 hover:text-slate-900'
          : 'bg-slate-800 border-slate-600 hover:bg-slate-700 text-slate-200 hover:text-slate-100'
      }`}
    >
      {theme === 'light' ? (
        <>
          <Moon className="w-4 h-4" />
          <span className="hidden sm:inline">Dark</span>
        </>
      ) : (
        <>
          <Sun className="w-4 h-4" />
          <span className="hidden sm:inline">Light</span>
        </>
      )}
    </Button>
  );
};