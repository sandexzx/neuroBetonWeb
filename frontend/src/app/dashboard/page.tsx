"use client";

import { ImageUploader } from '@/components/ImageUploader';
import { motion } from 'framer-motion';

export default function DashboardPage() {
  return (
    <div className="h-full flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
        className="w-full max-w-4xl px-8"
      >
        <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm rounded-2xl shadow-2xl p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-semibold tracking-tight mb-3 font-sans" style={{letterSpacing: '-0.02em'}}>
              Добро пожаловать в <span className="bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">Панель управления</span>
            </h1>
            <p className="text-lg text-muted-foreground font-sans">
              Загрузите изображение для анализа прочности бетона
            </p>
          </div>
          <ImageUploader />
        </div>
      </motion.div>
    </div>
  );
}
