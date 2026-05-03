"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/report", label: "Introduction", icon: "01" },
  { href: "/report/shock", label: "The Exogenous Shock", icon: "02" },
  { href: "/report/convergence", label: "Convergence & Stasis", icon: "03" },
  { href: "/report/education", label: "Connectivity & Education", icon: "04" },
  { href: "/report/equity", label: "The Equity Question", icon: "05" },
  { href: "/report/payments", label: "Digital Payments", icon: "06" },
  { href: "/report/divide", label: "Who's Left Behind", icon: "07" },
  { href: "/report/methodology", label: "Methodology", icon: "08" },
  { href: "/explorer", label: "Data Explorer", icon: "AI" },
];

export default function ReportNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 w-64 h-screen bg-[#292017] text-white flex flex-col overflow-y-auto z-50">
      <div className="p-6 border-b border-amber-900/40">
        <Link href="/report" className="block">
          <h1 className="text-lg font-bold tracking-tight text-white">
            India&apos;s Digital
            <br />
            Transformation
          </h1>
          <p className="text-xs text-amber-200/60 mt-1 font-medium">
            DSM Research Report 2026
          </p>
        </Link>
      </div>
      <div className="flex-1 py-4 px-3">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive =
              pathname === item.href ||
              (item.href !== "/report" && pathname?.startsWith(item.href + "/"));
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-amber-700 text-white shadow-lg shadow-amber-700/25"
                      : "text-amber-100/70 hover:bg-amber-900/30 hover:text-white"
                  }`}
                >
                  <span
                    className={`text-xs font-bold w-6 h-6 flex items-center justify-center rounded ${
                      isActive
                        ? "bg-amber-600 text-white"
                        : "bg-amber-900/40 text-amber-200/60"
                    }`}
                  >
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
      <div className="p-4 border-t border-amber-900/40">
        <p className="text-xs text-amber-200/50 text-center">
          Saumya Mishra &amp; Vatsl Goswami
        </p>
        <p className="text-xs text-amber-200/30 text-center mt-1">
          CS-3510 &middot; Data Science and Management
        </p>
      </div>
    </nav>
  );
}
