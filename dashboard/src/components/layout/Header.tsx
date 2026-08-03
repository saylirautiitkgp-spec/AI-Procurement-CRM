"use client";

import { Bell, Search } from "lucide-react";

export default function Header() {
  return (
    <header className="flex justify-between items-center bg-white shadow px-8 py-5 rounded-xl">

      <div className="flex items-center gap-3 bg-gray-100 px-4 py-3 rounded-lg w-[420px]">

        <Search size={18} />

        <input
          placeholder="Search Companies..."
          className="bg-transparent outline-none w-full"
        />

      </div>

      <div className="flex items-center gap-6">

        <Bell />

        <div className="font-bold">
          Sayli Raut
        </div>

      </div>

    </header>
  );
}