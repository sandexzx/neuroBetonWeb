"use client";

import { useRouter, usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { ArrowRightIcon, UserPlusIcon } from "@heroicons/react/24/outline";
import { motion } from "framer-motion";

export default function Home() {
  const router = useRouter();
  const pathname = usePathname();
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      const user = JSON.parse(savedUser);
      setUsername(user.username);
      if (pathname === '/') {
        router.push('/dashboard');
      }
    }
  }, [pathname]);

  if (!username) {
    return (
      <div className="flex items-center justify-center py-8">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: 'easeOut' }}
          className="w-full max-w-md"
        >
          <Card className="bg-gray-100 dark:bg-gray-900 shadow-none border-0">
            <div className="bg-white dark:bg-zinc-900 rounded-2xl shadow-2xl p-8 fade-in-up">
              <CardHeader className="text-center p-0 mb-4">
                <CardTitle className="text-4xl font-semibold tracking-tight mb-2 font-sans" style={{letterSpacing: '-0.02em'}}>
                  Welcome to <span className="bg-gradient-to-r from-black via-gray-700 to-gray-400 bg-clip-text text-transparent dark:from-white dark:via-gray-300 dark:to-gray-500">NeuroBeton</span>
                </CardTitle>
                <CardDescription className="text-lg text-muted-foreground mb-2 font-sans">
                  Analyze concrete strength using neural networks
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <div className="space-y-4">
                  <Button
                    onClick={() => router.push('/auth/login')}
                    variant="default"
                    size="lg"
                    className="w-full flex items-center justify-center gap-2 text-lg font-semibold shadow-md hover:scale-[1.03] active:scale-95 transition-transform duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20"
                  >
                    <ArrowRightIcon className="w-5 h-5" /> Login
                  </Button>
                  <Button
                    onClick={() => router.push('/auth/register')}
                    variant="outline"
                    size="lg"
                    className="w-full flex items-center justify-center gap-2 text-lg font-semibold shadow-md hover:scale-[1.03] active:scale-95 transition-transform duration-200 border-black/80 dark:border-white/10 hover:bg-black/10 dark:hover:bg-white/5"
                  >
                    <UserPlusIcon className="w-5 h-5" /> Register
                  </Button>
                </div>
              </CardContent>
            </div>
          </Card>
        </motion.div>
      </div>
    );
  }

  return null;
}
