import React from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useLanguage } from '@/contexts/LanguageContext';
import { Languages, Check } from 'lucide-react';

export const LanguageSwitcher: React.FC = () => {
  const { language, setLanguage, t } = useLanguage();

  const languages = [
    {
      value: 'en' as const,
      label: 'English',
      flag: '🇺🇸',
      nativeLabel: 'English'
    },
    {
      value: 'km' as const,
      label: 'Khmer',
      flag: '🇰🇭',
      nativeLabel: 'ខ្មែរ'
    }
  ];

  const currentLanguage = languages.find(l => l.value === language);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="flex items-center gap-2">
          <Languages className="h-4 w-4" />
          <span className="hidden sm:inline">
            {currentLanguage?.flag} {currentLanguage?.nativeLabel}
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        {languages.map((lang) => {
          const isActive = language === lang.value;
          
          return (
            <DropdownMenuItem
              key={lang.value}
              onClick={() => setLanguage(lang.value)}
              className="flex items-center justify-between cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <span className="text-lg">{lang.flag}</span>
                <div>
                  <div className="font-medium">{lang.nativeLabel}</div>
                  <div className="text-xs text-muted-foreground">
                    {lang.label}
                  </div>
                </div>
              </div>
              {isActive && <Check className="h-4 w-4" />}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};