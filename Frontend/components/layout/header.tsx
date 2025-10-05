'use client';
import { Satellite } from 'lucide-react';

type HeaderProps = {
    children?: React.ReactNode;
}

export function Header({ children }: HeaderProps) {
  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-20 flex h-16 items-center border-b border-primary/10 bg-background/50 px-4 backdrop-blur-md">
        <div className="flex w-full items-center justify-between">
          <div className="flex items-center gap-3">
              <Satellite className="h-7 w-7 text-primary" />
              <h1 className="text-xl font-bold tracking-tight text-foreground font-headline">
                ExoVerse Explorer
              </h1>
          </div>
          <div className="flex items-center gap-2">
            {children}
          </div>
        </div>
      </header>
    </>
  );
}
