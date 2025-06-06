"use client";

import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useEffect } from "react";
import { Navigation } from "./Navigation";

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
            <button
              onClick={handleLogout}
              className="text-destructive hover:text-destructive/80"
            >
              Logout
            </button>
          </div>
        )}
      </nav>
    </header>
  );
}
