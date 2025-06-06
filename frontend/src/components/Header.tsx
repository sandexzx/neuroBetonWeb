"use client";

import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useEffect } from "react";
import { Navigation } from "./Navigation";
import { Button } from "@/components/ui/button";
import { ArrowRightOnRectangleIcon } from "@heroicons/react/24/outline";

export default function Header() {
  const router = useRouter();
  const { user, logout } = useAuth();

  useEffect(() => {
    console.log('Current user:', user);
    console.log('localStorage user:', localStorage.getItem('user'));
  }, [user]);

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  return (
    <header className="bg-primary/10 p-4">
      <nav className="container mx-auto flex justify-between items-center">
        <div className="flex items-center gap-8">
          <h1 className="text-2xl font-bold">NeuroBeton</h1>
          {user && <Navigation />}
        </div>
        {user && (
          <div className="flex gap-4 items-center">
            <span className="text-muted-foreground">Welcome, {user.username}</span>
            <Button
              onClick={handleLogout}
              className="h-10 px-6 text-base font-semibold shadow-md hover:scale-[1.02] active:scale-95 transition-all duration-200 bg-black/80 dark:bg-white/10 hover:bg-black dark:hover:bg-white/20 rounded-lg flex items-center gap-2"
            >
              <ArrowRightOnRectangleIcon className="w-5 h-5" />
              Logout
            </Button>
          </div>
        )}
      </nav>
    </header>
  );
}
