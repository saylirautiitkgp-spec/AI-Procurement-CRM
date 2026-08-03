"use client";

import {
  LayoutDashboard,
  Building2,
  Truck,
  Users,
  Bot,
  Settings,
} from "lucide-react";

const menu = [
  { icon: LayoutDashboard, label: "Dashboard", active: true },
  { icon: Building2, label: "Companies" },
  { icon: Truck, label: "Suppliers" },
  { icon: Users, label: "Contacts" },
  { icon: Bot, label: "AI Copilot" },
  { icon: Settings, label: "Settings" },
];

export default function Sidebar() {
  return (
    <aside className="w-72 h-screen bg-white border-r border-gray-200 p-6">

      <h1 className="text-4xl font-extrabold text-[#D71920] mb-12">
        ProcureAI
      </h1>

      <div className="space-y-3">

        {menu.map((item) => (
          <button
            key={item.label}
            className={`flex items-center gap-4 w-full rounded-xl px-4 py-4 font-semibold transition
            ${
              item.active
                ? "bg-[#D71920] text-white shadow-lg"
                : "text-[#D71920] hover:bg-red-50"
            }`}
          >
            <item.icon size={22} />
            {item.label}
          </button>
        ))}

      </div>
    </aside>
  );
}