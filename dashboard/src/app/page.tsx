import Layout from "@/components/layout/Layout";
import KPICard from "@/components/dashboard/KPICard";
import AgentStatus from "@/components/dashboard/AgentStatus";
import RecentActivity from "@/components/dashboard/RecentActivity";

import {
    Building2,
    Truck,
    Users,
    Bot
} from "lucide-react";

export default function Home() {

    return (

        <Layout>

            <h1 className="text-4xl font-extrabold">
                Procurement Intelligence Dashboard
            </h1>

            <p className="text-gray-500 mb-8 mt-2">
                AI-powered Procurement CRM
            </p>

            <div className="grid grid-cols-4 gap-6">

                <KPICard
                    title="Companies"
                    value="482"
                    icon={Building2}
                />

                <KPICard
                    title="Suppliers"
                    value="176"
                    icon={Truck}
                />

                <KPICard
                    title="Contacts"
                    value="2381"
                    icon={Users}
                />

                <KPICard
                    title="AI Coverage"
                    value="94%"
                    icon={Bot}
                />

            </div>

            <div className="grid grid-cols-2 gap-6 mt-8">

                <AgentStatus />

                <RecentActivity />

            </div>

        </Layout>

    );

}