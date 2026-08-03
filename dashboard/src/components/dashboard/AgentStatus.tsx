import {
    CheckCircle2,
    LoaderCircle,
    Search,
    Database
} from "lucide-react";

const agents = [
    {
        name: "Company Search",
        status: "Completed",
        icon: Search,
        color: "text-green-600"
    },
    {
        name: "Website Verification",
        status: "Running",
        icon: LoaderCircle,
        color: "text-yellow-500 animate-spin"
    },
    {
        name: "Database Update",
        status: "Pending",
        icon: Database,
        color: "text-gray-400"
    },
    {
        name: "Contact Discovery",
        status: "Completed",
        icon: CheckCircle2,
        color: "text-green-600"
    }
];

export default function AgentStatus() {
    return (

        <div className="bg-white rounded-2xl shadow p-6">

            <h2 className="text-xl font-bold mb-5">
                AI Workflow
            </h2>

            <div className="space-y-4">

                {agents.map((agent) => (

                    <div
                        key={agent.name}
                        className="flex justify-between items-center"
                    >

                        <div className="flex gap-3 items-center">

                            <agent.icon
                                className={agent.color}
                                size={22}
                            />

                            <span>{agent.name}</span>

                        </div>

                        <span className="text-gray-500">
                            {agent.status}
                        </span>

                    </div>

                ))}

            </div>

        </div>

    );
}