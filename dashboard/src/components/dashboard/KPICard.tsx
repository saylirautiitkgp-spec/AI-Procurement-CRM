import { LucideIcon } from "lucide-react";

interface Props {
  title: string;
  value: string;
  icon: LucideIcon;
}

export default function KPICard({
  title,
  value,
  icon: Icon,
}: Props) {
  return (
    <div className="bg-white rounded-2xl shadow p-6 flex justify-between items-center">

      <div>

        <p className="text-gray-500 text-sm">
          {title}
        </p>

        <h2 className="text-3xl font-extrabold mt-2">
          {value}
        </h2>

      </div>

      <div className="bg-red-100 p-4 rounded-full">

        <Icon
          className="text-red-600"
          size={28}
        />

      </div>

    </div>
  );
}