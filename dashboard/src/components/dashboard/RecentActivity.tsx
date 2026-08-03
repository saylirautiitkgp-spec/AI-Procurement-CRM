const activities = [

    "Bosch India analyzed",

    "Siemens supplier portal found",

    "ABB contact discovered",

    "Motherson workflow completed",

    "Tata Motors updated"

];

export default function RecentActivity() {

    return (

        <div className="bg-white rounded-2xl shadow p-6">

            <h2 className="text-xl font-bold mb-5">
                Recent Activity
            </h2>

            <div className="space-y-4">

                {activities.map((item) => (

                    <div
                        key={item}
                        className="border-b pb-3"
                    >
                        {item}
                    </div>

                ))}

            </div>

        </div>

    );

}