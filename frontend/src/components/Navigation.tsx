"use client"

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { HomeIcon, ClockIcon } from "@heroicons/react/24/outline";
import { cn } from '@/lib/utils';

export function Navigation() {
  const pathname = usePathname();
  
  // Показываем навигацию только на страницах дашборда и истории
  if (!['/dashboard', '/history'].includes(pathname)) {
    return null;
  }

  const navItems = [
    {
      href: '/dashboard',
      label: 'Панель управления',
      icon: HomeIcon,
    },
    {
      href: '/history', 
      label: 'История',
      icon: ClockIcon,
    },
  ];

  return (
    <nav className="flex items-center gap-1 bg-white/20 dark:bg-zinc-900/20 backdrop-blur-sm rounded-xl p-1 shadow-sm">
      {navItems.map((item) => {
        const isActive = pathname === item.href;
        const Icon = item.icon;
        
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
              isActive
                ? "text-black dark:text-white"
                : "text-gray-600 dark:text-gray-400 hover:text-black dark:hover:text-white"
            )}
          >
            {/* Активный индикатор с анимацией */}
            {isActive && (
              <motion.div
                layoutId="activeTab"
                className="absolute inset-0 bg-white/50 dark:bg-white/10 rounded-lg shadow-sm"
                initial={false}
                transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
              />
            )}
            
            {/* Иконка и текст */}
            <div className="relative z-10 flex items-center gap-2">
              <Icon className="w-4 h-4" />
              <span>{item.label}</span>
            </div>
          </Link>
        );
      })}
    </nav>
  );
}