import Sidebar from "./Sidebar";
import Header from "./Header";

export default function Layout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex bg-slate-100">

      <Sidebar />

      <main className="flex-1 p-8">

        <Header />

        <div className="mt-8">
          {children}
        </div>

      </main>

    </div>
  );
}