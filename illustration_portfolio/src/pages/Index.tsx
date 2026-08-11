import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogTrigger } from '@/components/ui/dialog';
import { ChevronLeft, ChevronRight, X, Filter, Calendar, DollarSign, BookOpen, Heart, Dumbbell, Target, Clock } from 'lucide-react';

// Types
interface PortfolioItem {
  id: string;
  title: string;
  series: 'Commercial' | 'Personal Work';
  image: string;
  description: string;
  year: string;
}

interface JournalEntry {
  id: string;
  date: string;
  completed: string;
  todo: string;
  inProgress: string;
  exercised: boolean;
  spending: number;
  learned: string;
  emotion: string;
  grateful: string;
}

// Sample portfolio data
const portfolioItems: PortfolioItem[] = [
  {
    id: '1',
    title: 'Brand Identity Design',
    series: 'Commercial',
    image: '/images/portfolio_commercial_1.png',
    description: 'Complete brand identity design for a tech startup',
    year: '2024'
  },
  {
    id: '2',
    title: 'Digital Portrait Series',
    series: 'Personal Work',
    image: '/images/portfolio_personal_1.jpeg',
    description: 'Experimental digital portraits exploring emotion',
    year: '2024'
  },
  {
    id: '3',
    title: 'Product Packaging',
    series: 'Commercial',
    image: '/images/portfolio_commercial_2.png',
    description: 'Packaging design for organic skincare products',
    year: '2023'
  },
  {
    id: '4',
    title: 'Abstract Compositions',
    series: 'Personal Work',
    image: '/images/portfolio_personal_2.jpeg',
    description: 'Abstract digital art exploring color and form',
    year: '2023'
  },
  {
    id: '5',
    title: 'Website Illustrations',
    series: 'Commercial',
    image: '/images/portfolio_commercial_3.png',
    description: 'Custom illustrations for web applications',
    year: '2024'
  },
  {
    id: '6',
    title: 'Character Design',
    series: 'Personal Work',
    image: '/images/portfolio_personal_3.png',
    description: 'Original character designs and concepts',
    year: '2024'
  },
  {
    id: '7',
    title: 'Editorial Illustrations',
    series: 'Commercial',
    image: '/images/portfolio_commercial_4.jpeg',
    description: 'Magazine and editorial illustrations',
    year: '2023'
  },
  {
    id: '8',
    title: 'Book Cover Design',
    series: 'Commercial',
    image: '/images/portfolio_commercial_5.png',
    description: 'Book cover design for fiction novel',
    year: '2024'
  }
];

// Emotion stickers data
const emotionStickers = [
  { emoji: '😊', label: 'Happy' },
  { emoji: '😔', label: 'Sad' },
  { emoji: '😴', label: 'Tired' },
  { emoji: '🤔', label: 'Thoughtful' },
  { emoji: '😤', label: 'Frustrated' },
  { emoji: '🥳', label: 'Excited' },
  { emoji: '😌', label: 'Peaceful' },
  { emoji: '🤗', label: 'Grateful' },
  { emoji: '💪', label: 'Motivated' },
  { emoji: '🎨', label: 'Creative' }
];

export default function Index() {
  const [selectedSeries, setSelectedSeries] = useState<string>('All');
  const [lightboxImage, setLightboxImage] = useState<PortfolioItem | null>(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);
  const [currentEntry, setCurrentEntry] = useState<JournalEntry>({
    id: '',
    date: new Date().toISOString().split('T')[0],
    completed: '',
    todo: '',
    inProgress: '',
    exercised: false,
    spending: 0,
    learned: '',
    emotion: '',
    grateful: ''
  });

  // Load journal entries from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('journalEntries');
    if (saved) {
      setJournalEntries(JSON.parse(saved));
    }
  }, []);

  // Save journal entries to localStorage
  const saveJournalEntry = () => {
    const entry = {
      ...currentEntry,
      id: currentEntry.id || Date.now().toString()
    };
    
    const existingIndex = journalEntries.findIndex(e => e.date === entry.date);
    let updatedEntries;
    
    if (existingIndex >= 0) {
      updatedEntries = [...journalEntries];
      updatedEntries[existingIndex] = entry;
    } else {
      updatedEntries = [...journalEntries, entry];
    }
    
    setJournalEntries(updatedEntries);
    localStorage.setItem('journalEntries', JSON.stringify(updatedEntries));
  };

  // Load journal entry for selected date
  const loadJournalEntry = (date: string) => {
    const entry = journalEntries.find(e => e.date === date);
    if (entry) {
      setCurrentEntry(entry);
    } else {
      setCurrentEntry({
        id: '',
        date,
        completed: '',
        todo: '',
        inProgress: '',
        exercised: false,
        spending: 0,
        learned: '',
        emotion: '',
        grateful: ''
      });
    }
  };

  // Filter portfolio items
  const filteredItems = selectedSeries === 'All' 
    ? portfolioItems 
    : portfolioItems.filter(item => item.series === selectedSeries);

  // Lightbox navigation
  const openLightbox = (item: PortfolioItem) => {
    setLightboxImage(item);
    setCurrentImageIndex(filteredItems.findIndex(i => i.id === item.id));
  };

  const navigateLightbox = (direction: 'prev' | 'next') => {
    const newIndex = direction === 'prev' 
      ? (currentImageIndex - 1 + filteredItems.length) % filteredItems.length
      : (currentImageIndex + 1) % filteredItems.length;
    
    setCurrentImageIndex(newIndex);
    setLightboxImage(filteredItems[newIndex]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50">
      <div className="flex">
        {/* Weekly Review Sidebar */}
        <div className="w-80 min-h-screen bg-white/80 backdrop-blur-sm border-r border-purple-200 p-6 overflow-y-auto">
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-purple-800 mb-2">Weekly Review</h2>
              <p className="text-sm text-purple-600">Track your daily journey</p>
            </div>

            {/* Date Selector */}
            <div className="journal-card">
              <label className="block text-sm font-medium text-purple-700 mb-2">
                <Calendar className="inline w-4 h-4 mr-1" />
                Select Date
              </label>
              <Input
                type="date"
                value={currentEntry.date}
                onChange={(e) => {
                  const newDate = e.target.value;
                  setCurrentEntry(prev => ({ ...prev, date: newDate }));
                  loadJournalEntry(newDate);
                }}
                className="w-full"
              />
            </div>

            {/* Journal Form */}
            <div className="space-y-4">
              {/* What's Done */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-2">
                  <Target className="inline w-4 h-4 mr-1" />
                  What has been done?
                </label>
                <Textarea
                  value={currentEntry.completed}
                  onChange={(e) => setCurrentEntry(prev => ({ ...prev, completed: e.target.value }))}
                  placeholder="List your accomplishments..."
                  className="min-h-[80px]"
                />
              </div>

              {/* To Do Today */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-2">
                  <Clock className="inline w-4 h-4 mr-1" />
                  What to do today?
                </label>
                <Textarea
                  value={currentEntry.todo}
                  onChange={(e) => setCurrentEntry(prev => ({ ...prev, todo: e.target.value }))}
                  placeholder="Plan your day..."
                  className="min-h-[80px]"
                />
              </div>

              {/* In Progress */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-2">
                  What is in progress?
                </label>
                <Textarea
                  value={currentEntry.inProgress}
                  onChange={(e) => setCurrentEntry(prev => ({ ...prev, inProgress: e.target.value }))}
                  placeholder="Ongoing projects..."
                  className="min-h-[60px]"
                />
              </div>

              {/* Exercise */}
              <div className="journal-card">
                <label className="flex items-center space-x-2">
                  <Dumbbell className="w-4 h-4 text-purple-700" />
                  <span className="text-sm font-medium text-purple-700">Did I exercise today?</span>
                  <input
                    type="checkbox"
                    checked={currentEntry.exercised}
                    onChange={(e) => setCurrentEntry(prev => ({ ...prev, exercised: e.target.checked }))}
                    className="ml-auto"
                  />
                </label>
              </div>

              {/* Spending */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-2">
                  <DollarSign className="inline w-4 h-4 mr-1" />
                  How much did I spend today?
                </label>
                <Input
                  type="number"
                  value={currentEntry.spending}
                  onChange={(e) => setCurrentEntry(prev => ({ ...prev, spending: parseFloat(e.target.value) || 0 }))}
                  placeholder="0.00"
                  step="0.01"
                />
              </div>

              {/* Learning */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-2">
                  <BookOpen className="inline w-4 h-4 mr-1" />
                  What did I learn today?
                </label>
                <Textarea
                  value={currentEntry.learned}
                  onChange={(e) => setCurrentEntry(prev => ({ ...prev, learned: e.target.value }))}
                  placeholder="New insights, skills, or knowledge..."
                  className="min-h-[60px]"
                />
              </div>

              {/* Emotion Stickers */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-3">
                  Emotion of the day
                </label>
                <div className="grid grid-cols-5 gap-2">
                  {emotionStickers.map((sticker) => (
                    <button
                      key={sticker.label}
                      onClick={() => setCurrentEntry(prev => ({ ...prev, emotion: sticker.emoji }))}
                      className={`p-2 rounded-lg text-2xl hover:bg-purple-100 transition-colors ${
                        currentEntry.emotion === sticker.emoji ? 'bg-purple-200 ring-2 ring-purple-400' : ''
                      }`}
                      title={sticker.label}
                    >
                      {sticker.emoji}
                    </button>
                  ))}
                </div>
              </div>

              {/* Gratitude */}
              <div className="journal-card">
                <label className="block text-sm font-medium text-purple-700 mb-2">
                  <Heart className="inline w-4 h-4 mr-1" />
                  What am I most grateful for?
                </label>
                <Textarea
                  value={currentEntry.grateful}
                  onChange={(e) => setCurrentEntry(prev => ({ ...prev, grateful: e.target.value }))}
                  placeholder="Express your gratitude..."
                  className="min-h-[60px]"
                />
              </div>

              {/* Save Button */}
              <Button 
                onClick={saveJournalEntry}
                className="w-full creative-gradient text-white hover:opacity-90"
              >
                Save Journal Entry
              </Button>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-8">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              Creative Portfolio
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Showcasing my artistic journey through digital illustrations, brand designs, and personal creative explorations.
            </p>
          </div>

          {/* Portfolio Filter */}
          <div className="flex justify-center mb-8">
            <div className="flex items-center space-x-4 bg-white/80 backdrop-blur-sm rounded-full p-2 shadow-soft">
              <Filter className="w-5 h-5 text-purple-600 ml-4" />
              {['All', 'Commercial', 'Personal Work'].map((series) => (
                <Button
                  key={series}
                  variant={selectedSeries === series ? "default" : "ghost"}
                  onClick={() => setSelectedSeries(series)}
                  className={selectedSeries === series ? "creative-gradient text-white" : ""}
                >
                  {series}
                </Button>
              ))}
            </div>
          </div>

          {/* Portfolio Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {filteredItems.map((item, index) => (
              <div
                key={item.id}
                className="portfolio-card animate-fade-in-up"
                style={{ animationDelay: `${index * 0.1}s` }}
                onClick={() => openLightbox(item)}
              >
                <div className="aspect-[4/3] overflow-hidden">
                  <img
                    src={item.image}
                    alt={item.title}
                    className="w-full h-full object-cover hover:scale-105 transition-transform duration-500"
                  />
                </div>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="secondary" className="text-xs">
                      {item.series}
                    </Badge>
                    <span className="text-sm text-gray-500">{item.year}</span>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{item.title}</h3>
                  <p className="text-gray-600 text-sm">{item.description}</p>
                </div>
              </div>
            ))}
          </div>

          {/* About Section */}
          <div className="max-w-4xl mx-auto text-center">
            <Card className="glass-effect border-purple-200">
              <CardContent className="p-8">
                <h2 className="text-3xl font-bold text-purple-800 mb-4">About My Work</h2>
                <p className="text-gray-700 leading-relaxed mb-6">
                  I'm a digital illustrator passionate about creating meaningful visual experiences. 
                  My work spans commercial projects and personal explorations, always seeking to 
                  push creative boundaries and tell compelling stories through art.
                </p>
                <div className="flex justify-center space-x-4">
                  <Badge className="creative-gradient text-white">Digital Illustration</Badge>
                  <Badge className="warm-gradient text-white">Brand Design</Badge>
                  <Badge className="cool-gradient text-white">Creative Direction</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Lightbox Dialog */}
      <Dialog open={!!lightboxImage} onOpenChange={() => setLightboxImage(null)}>
        <DialogContent className="max-w-6xl w-full h-[90vh] p-0 bg-black/95">
          {lightboxImage && (
            <div className="relative w-full h-full flex items-center justify-center">
              {/* Close Button */}
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-4 right-4 z-10 text-white hover:bg-white/20"
                onClick={() => setLightboxImage(null)}
              >
                <X className="w-6 h-6" />
              </Button>

              {/* Navigation Buttons */}
              <Button
                variant="ghost"
                size="icon"
                className="absolute left-4 top-1/2 -translate-y-1/2 z-10 text-white hover:bg-white/20"
                onClick={() => navigateLightbox('prev')}
              >
                <ChevronLeft className="w-8 h-8" />
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className="absolute right-4 top-1/2 -translate-y-1/2 z-10 text-white hover:bg-white/20"
                onClick={() => navigateLightbox('next')}
              >
                <ChevronRight className="w-8 h-8" />
              </Button>

              {/* Image */}
              <div className="w-full h-full flex items-center justify-center p-8">
                <img
                  src={lightboxImage.image}
                  alt={lightboxImage.title}
                  className="max-w-full max-h-full object-contain animate-scale-in"
                />
              </div>

              {/* Image Info */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-8">
                <div className="text-white">
                  <div className="flex items-center space-x-4 mb-2">
                    <Badge className="bg-white/20 text-white">
                      {lightboxImage.series}
                    </Badge>
                    <span className="text-sm opacity-80">{lightboxImage.year}</span>
                  </div>
                  <h3 className="text-2xl font-bold mb-2">{lightboxImage.title}</h3>
                  <p className="text-gray-300">{lightboxImage.description}</p>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}