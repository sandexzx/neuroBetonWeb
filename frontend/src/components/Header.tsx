"use client";

import { useRouter, usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { ArrowRightOnRectangleIcon } from "@heroicons/react/24/outline";

export default function Header() {
  const router = useRouter();
  const pathname = usePathname();
  const { user, logout } = useAuth();

  useEffect(() => {
    console.log('Header - Current user state:', user);
    console.log('Header - localStorage user:', localStorage.getItem('user'));
  }, [user]);

  useEffect(() => {
    // Не делаем редирект, если мы уже на странице логина или регистрации
    if (!user && !pathname?.startsWith('/auth/')) {
      console.log('Header - No user found, redirecting to login');
      router.push('/auth/login');
    }
  }, [user, router, pathname]);

  const handleLogout = () => {
    console.log('Header - Logging out user:', user);
    logout();
    router.push('/');
  };

  // Если пользователь не авторизован, показываем только логотип
  if (!user) {
    return (
      <header className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm border-b border-black/5 dark:border-white/5">
        <nav className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <motion.h1 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-2xl font-bold tracking-tight bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
            >
              NeuroBeton
            </motion.h1>
          </div>
        </nav>
      </header>
    );
  }

  return (
    <header className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-sm border-b border-black/5 dark:border-white/5">
      <nav className="container mx-auto px-6 py-4">
        <div className="flex justify-between items-center">
          <motion.h1 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl font-bold tracking-tight bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500"
          >
            NeuroBeton
          </motion.h1>
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-8"
          >
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="text-base font-medium tracking-wide"
            >
              <span className="text-muted-foreground">Welcome, </span>
              <span className="font-semibold text-primary dark:text-primary/90">
                {user.username}
              </span>
            </motion.div>
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <Button
                onClick={handleLogout}
                className="h-10 px-6 text-base font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20 rounded-lg flex items-center gap-2"
              >
                <ArrowRightOnRectangleIcon className="w-5 h-5" />
                Logout
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </nav>
    </header>
  );
}
