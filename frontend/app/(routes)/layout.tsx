import SideNav from "@/components/SideNav";

export default async function SetupLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="grid h-screen w-full pl-[56px]">
      <SideNav />
      <div className="flex flex-col">
        <header className="sticky top-0 z-10 flex h-[57px] items-center gap-1 border-b bg-background px-4">
          <h1 className="text-xl font-semibold">UNSW MOGONET</h1>
        </header>
        <main className="grid flex-1 gap-4 overflow-auto p-4 grid-cols-6">
          {children}
        </main>
      </div>
    </div>
  );
}
