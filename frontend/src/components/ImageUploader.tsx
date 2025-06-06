"use client"

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowUpTrayIcon, XMarkIcon, PhotoIcon } from "@heroicons/react/24/outline";
import { useAuth } from '@/lib/auth';
import { motion, AnimatePresence } from 'framer-motion';
import { useToast } from '@/components/ui/toast';

interface PredictionResult {
  strength: number;
  cracks: {
    has_cracks: boolean;
    probability: number;
  };
  concrete_type: {
    type: string;
    confidence: number;
  };
}

export function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const { user } = useAuth();
  const { addToast } = useToast();

  const handleFileChange = useCallback((file: File) => {
    // Валидация типа файла
    if (!file.type.startsWith('image/')) {
      addToast({
        type: 'error',
        title: 'Invalid file type',
        description: 'Please select an image file (JPG, PNG, etc.)'
      });
      return;
    }
    
    // Валидация размера файла (максимум 10MB)
    if (file.size > 10 * 1024 * 1024) {
      addToast({
        type: 'error',
        title: 'File too large',
        description: 'Please select an image smaller than 10MB'
      });
      return;
    }

    setSelectedFile(file);
    setPrediction(null);
    
    // Создаем превью изображения
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, [addToast]);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      handleFileChange(event.target.files[0]);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      handleFileChange(imageFile);
    }
  }, [handleFileChange]);

  const removeImage = useCallback(() => {
    setSelectedFile(null);
    setImagePreview(null);
    setPrediction(null);
  }, []);

  const handleSubmit = async () => {
    if (!selectedFile || !user) return;

    setLoading(true);
    setPrediction(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`http://localhost:8000/predict?username=${user.username}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to analyze image');
      }

      const data = await response.json();
      setPrediction(data);
      addToast({
        type: 'success',
        title: 'Analysis complete!',
        description: `Concrete strength: ${data.strength.toFixed(2)} MPa`
      });
    } catch (error: any) {
      console.error('Error:', error);
      addToast({
        type: 'error',
        title: 'Analysis failed',
        description: error.message || 'Please try again with a different image.'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full">
      <div className="space-y-8">
        {/* Зона загрузки файлов */}
        <AnimatePresence mode="wait">
          {!imagePreview ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3 }}
              className={`
                relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300
                ${isDragOver 
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/20 scale-[1.02]' 
                  : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                }
                ${loading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
              `}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !loading && document.getElementById('file-upload')?.click()}
            >
              <motion.div
                animate={{ 
                  y: isDragOver ? -5 : 0,
                  scale: isDragOver ? 1.1 : 1 
                }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
              >
                <PhotoIcon className="mx-auto h-16 w-16 text-gray-400 mb-4" />
              </motion.div>
              
              <h3 className="text-xl font-semibold mb-2 text-gray-700 dark:text-gray-300">
                {isDragOver ? 'Drop your image here!' : 'Upload Concrete Image'}
              </h3>
              <p className="text-gray-500 dark:text-gray-400 mb-6">
                Drag and drop an image, or click to select
              </p>
              
              <Input
                type="file"
                accept="image/*"
                onChange={handleInputChange}
                disabled={loading}
                className="hidden"
                id="file-upload"
              />
              
              <Button
                type="button"
                disabled={loading}
                className="h-12 px-8 text-lg font-semibold shadow-lg hover:scale-[1.02] active:scale-95 transition-all duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20 rounded-xl"
              >
                <ArrowUpTrayIcon className="w-6 h-6 mr-3" />
                Choose File
              </Button>
            </motion.div>
          ) : (
            /* Превью загруженного изображения */
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="relative"
            >
              <div className="relative bg-white dark:bg-zinc-900 rounded-2xl p-6 shadow-xl border border-gray-200 dark:border-gray-700">
                <div className="flex items-start gap-6">
                  {/* Превью изображения */}
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="relative flex-shrink-0"
                  >
                    <div className="relative w-32 h-32 sm:w-40 sm:h-40 rounded-xl overflow-hidden shadow-lg">
                      <img
                        src={imagePreview}
                        alt="Uploaded concrete sample"
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
                    </div>
                    
                    {/* Кнопка удаления - более деликатная */}
                    <motion.button
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 0.7 }}
                      whileHover={{ opacity: 1, scale: 1.1 }}
                      transition={{ delay: 0.3, type: "spring", stiffness: 300 }}
                      onClick={removeImage}
                      className="absolute -top-2 -right-2 w-7 h-7 bg-gray-800/80 dark:bg-gray-200/80 text-white dark:text-gray-800 rounded-full shadow-md backdrop-blur-sm hover:bg-gray-900/90 dark:hover:bg-gray-100/90 transition-all duration-200 flex items-center justify-center"
                    >
                      <XMarkIcon className="w-3.5 h-3.5" />
                    </motion.button>
                  </motion.div>

                  {/* Информация о файле */}
                  <motion.div
                    initial={{ x: 20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.4, delay: 0.2 }}
                    className="flex-1 min-w-0"
                  >
                    <div className="mb-4">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate">
                        {selectedFile?.name}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        {selectedFile && `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`}
                      </p>
                    </div>

                    {/* Кнопки действий */}
                    <div className="flex flex-col sm:flex-row gap-3">
                      <Button
                        onClick={handleSubmit}
                        disabled={loading || !user}
                        className="flex-1 h-11 text-base font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 bg-green-600 hover:bg-green-700 dark:bg-green-700 dark:hover:bg-green-600 rounded-xl"
                      >
                        {loading ? (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            className="w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2"
                          />
                        ) : (
                          <ArrowUpTrayIcon className="w-5 h-5 mr-2" />
                        )}
                        {loading ? 'Analyzing...' : 'Analyze Image'}
                      </Button>
                      
                      <Button
                        onClick={() => document.getElementById('file-upload')?.click()}
                        variant="outline"
                        disabled={loading}
                        className="h-11 px-6 text-base font-medium hover:scale-[1.02] active:scale-95 transition-all duration-200 rounded-xl border-2 border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800/50 text-gray-600 dark:text-gray-300"
                      >
                        <PhotoIcon className="w-4 h-4 mr-2" />
                        Replace
                      </Button>
                    </div>
                  </motion.div>
                </div>
              </div>
              
              {/* Скрытый input для смены файла */}
              <Input
                type="file"
                accept="image/*"
                onChange={handleInputChange}
                disabled={loading}
                className="hidden"
                id="file-upload"
              />
            </motion.div>
          )}
        </AnimatePresence>

        {prediction !== null && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ 
              duration: 0.6, 
              ease: [0.25, 0.46, 0.45, 0.94],
              staggerChildren: 0.2
            }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {/* Strength Prediction */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.4 }}
              className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm border border-white/20 dark:border-white/10 flex flex-col items-center"
            >
              <motion.h3 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2, duration: 0.4 }}
                className="text-lg font-semibold mb-4 text-muted-foreground text-center w-full"
              >
                Analysis Complete
              </motion.h3>
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ 
                  delay: 0.4, 
                  duration: 0.5, 
                  type: "spring", 
                  stiffness: 200 
                }}
                className="text-center"
              >
                <motion.span 
                  className="text-5xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
                  initial={{ backgroundPosition: "0% 50%" }}
                  animate={{ backgroundPosition: "100% 50%" }}
                  transition={{ 
                    duration: 2, 
                    repeat: Infinity, 
                    repeatType: "reverse",
                    ease: "easeInOut"
                  }}
                  style={{ backgroundSize: "200% 200%" }}
                >
                  {prediction.strength.toFixed(2)}
                </motion.span>
                <motion.span 
                  className="text-2xl font-semibold text-muted-foreground ml-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8, duration: 0.3 }}
                >
                  MPa
                </motion.span>
              </motion.div>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ delay: 0.6, duration: 0.8, ease: "easeOut" }}
                className="h-1 bg-gradient-to-r from-emerald-500 via-blue-500 to-purple-500 rounded-full mt-4 mx-auto"
              />
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.4 }}
                className="text-center text-sm text-muted-foreground mt-3"
              >
                Concrete strength prediction
              </motion.p>
            </motion.div>

            {/* Cracks Detection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.4 }}
              className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm border border-white/20 dark:border-white/10 flex flex-col items-center"
            >
              <motion.h3 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.4 }}
                className="text-lg font-semibold mb-4 text-muted-foreground text-center w-full"
              >
                Cracks Detection
              </motion.h3>
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5, duration: 0.5 }}
                className="text-center"
              >
                <motion.p 
                  className="text-2xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6, duration: 0.4 }}
                >
                  {prediction.cracks.has_cracks ? 'Cracks Detected' : 'No Cracks'}
                </motion.p>
                <motion.p 
                  className="text-sm text-muted-foreground mt-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.7, duration: 0.4 }}
                >
                  Confidence: {(prediction.cracks.probability * 100).toFixed(1)}%
                </motion.p>
              </motion.div>
            </motion.div>

            {/* Concrete Type */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.4 }}
              className="p-8 bg-gradient-to-br from-black/5 to-black/10 dark:from-white/5 dark:to-white/10 rounded-2xl shadow-sm backdrop-blur-sm border border-white/20 dark:border-white/10 flex flex-col items-center"
            >
              <motion.h3 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.4 }}
                className="text-lg font-semibold mb-4 text-muted-foreground text-center w-full"
              >
                Concrete Type
              </motion.h3>
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6, duration: 0.5 }}
                className="text-center"
              >
                <motion.p 
                  className="text-2xl font-bold bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.7, duration: 0.4 }}
                >
                  {prediction.concrete_type.type}
                </motion.p>
                <motion.p 
                  className="text-sm text-muted-foreground mt-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8, duration: 0.4 }}
                >
                  Confidence: {(prediction.concrete_type.confidence * 100).toFixed(1)}%
                </motion.p>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
}