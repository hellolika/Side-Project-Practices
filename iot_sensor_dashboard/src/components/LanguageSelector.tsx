import React from 'react';
import { Button } from '@/components/ui/button';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import { Languages } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';
import { useTheme } from '@/contexts/ThemeContext';
import { Language } from '@/types/language';

const languageOptions = [
  { code: 'en' as Language, name: 'English', flag: '🇺🇸' },
  { code: 'km' as Language, name: 'ខ្មែរ', flag: '🇰🇭' },
];

export const LanguageSelector: React.FC = () => {
  const { language, setLanguage } = useLanguage();
  const { theme } = useTheme();

  const currentLanguage = languageOptions.find(lang => lang.code === language);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="outline" 
          size="sm" 
          className={`gap-2 transition-colors duration-200 ${
            theme === 'light'
              ? 'bg-slate-100 border-slate-300 hover:bg-slate-200 text-slate-700 hover:text-slate-900'
              : 'bg-slate-800 border-slate-600 hover:bg-slate-700 text-slate-200 hover:text-slate-100'
          }`}
        >
          <Languages className="w-4 h-4" />
          <span className="hidden sm:inline">{currentLanguage?.flag}</span>
          <span className="hidden md:inline">{currentLanguage?.name}</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent 
        align="end" 
        className={`transition-colors duration-200 ${
          theme === 'light'
            ? 'bg-white border-slate-200'
            : 'bg-slate-800 border-slate-600'
        }`}
      >
        {languageOptions.map((lang) => (
          <DropdownMenuItem
            key={lang.code}
            onClick={() => setLanguage(lang.code)}
            className={`cursor-pointer transition-colors duration-200 ${
              theme === 'light'
                ? `hover:bg-slate-100 text-slate-700 ${
                    language === lang.code ? 'bg-slate-100' : ''
                  }`
                : `hover:bg-slate-700 text-slate-200 ${
                    language === lang.code ? 'bg-slate-700' : ''
                  }`
            }`}
          >
            <span className="mr-2">{lang.flag}</span>
            <span>{lang.name}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};