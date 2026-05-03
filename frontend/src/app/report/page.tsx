import StatTable from "@/components/StatTable";

export default function IntroductionPage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">
          India&apos;s Digital Transformation
        </h1>
        <p>
          A Quantitative Analysis of Telecommunications, Digital Payments, and
          Socioeconomic Impact (2009-2023)
        </p>
        <p className="!text-amber-200/80 text-sm mt-3">
          Saumya Mishra &amp; Vatsl Goswami &mdash; CS-3510: Data Science and Management
        </p>
      </div>

      <blockquote>
        In September 2016, Reliance Jio launched commercial operations with an
        unprecedented offer: free 4G data for all. Within six months, India
        added more wireless subscribers than the entire population of Germany.
        This report investigates whether that exogenous shock merely changed
        who holds a SIM card&mdash;or whether it fundamentally altered the
        trajectory of human development across Indian states.
      </blockquote>

      <h2>Key Findings</h2>

      <div className="not-prose grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 my-8">
        <div className="metric-card">
          <div className="metric-value">253.60</div>
          <div className="metric-label">Sup-F Statistic (Critical: 12.35)</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">p = 0.003</div>
          <div className="metric-label">Pre-Jio Convergence (HC3)</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">p = 0.667</div>
          <div className="metric-label">Post-Jio Convergence (Stasis)</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">r = 0.77</div>
          <div className="metric-label">Cross-State Teledensity-GER</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">~135 yrs</div>
          <div className="metric-label">Bihar Catch-Up Gap</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">HHI: 2500 to 1800</div>
          <div className="metric-label">Market Concentration Shift</div>
        </div>
      </div>

      <h2>Research Questions</h2>

      <ol>
        <li>
          <strong>Structural Break Detection:</strong> Can we formally identify
          Jio&apos;s entry as a structural break in national wireless subscriber
          growth? (Sup-Wald test with heteroskedasticity-robust inference)
        </li>
        <li>
          <strong>Connectivity-Education Nexus:</strong> Does state-level
          teledensity correlate with or cause changes in Gross Enrolment Ratio?
          (Panel fixed-effects with two-way clustering)
        </li>
        <li>
          <strong>Digital Payments Revolution:</strong> How has the payments
          landscape shifted, and does mobile connectivity Granger-cause digital
          transaction growth?
        </li>
        <li>
          <strong>The Digital Divide:</strong> Using PCA and Louvain community
          detection, which states form persistent &quot;digital clusters,&quot;
          and how wide is the gap?
        </li>
        <li>
          <strong>SC/ST Equity:</strong> Has the expansion of
          telecommunications improved educational inclusion for marginalized
          communities?
        </li>
        <li>
          <strong>Convergence vs. Stasis:</strong> Were Indian states converging
          in teledensity before Jio, and did that convergence continue or halt
          after the shock?
        </li>
      </ol>

      <h2>Data Sources</h2>

      <StatTable
        title="Primary Data Sources"
        rows={[
          { label: "TRAI Telecom Subscription Reports", value: "2009-2023, monthly" },
          { label: "AISHE Higher Education Data", value: "2012-2022, annual" },
          { label: "RBI Digital Payments Statistics", value: "2017-2022, monthly" },
          { label: "UDISE School Education Data", value: "2013-2022, annual" },
          { label: "Census & State Population Estimates", value: "2011-2021" },
        ]}
      />

      <h2>Methodology Summary</h2>

      <p>
        This study employs a multi-method approach combining time-series
        econometrics, panel data analysis, and network-based clustering. Each
        finding is subjected to multiple robustness checks including
        heteroskedasticity-consistent (HC3) standard errors, Newey-West
        corrections for serial correlation, bootstrap resampling, and
        non-parametric alternatives. We adopt a deliberately conservative
        interpretive framework: results are reported as &quot;statistically
        significant&quot; only when they survive all robustness checks.
      </p>

      <p>
        The analysis pipeline is fully reproducible, built in Python with
        statsmodels, scipy, scikit-learn, and networkx. All raw data is sourced
        from official government repositories (TRAI, AISHE, RBI). The
        interactive charts in this report allow direct exploration of the
        underlying data.
      </p>


      <h2>Report Structure</h2>

      <p>
        The report proceeds through seven substantive chapters. We begin with
        the <strong>Exogenous Shock</strong>&mdash;formally establishing Jio&apos;s
        entry as a structural break. We then present our strongest finding:
        <strong> Convergence &amp; Stasis</strong>, showing that states were
        converging in teledensity before Jio but ceased converging after. The
        remaining chapters explore downstream effects on education, equity,
        digital payments, and the persistent digital divide. We conclude with a
        full methodological appendix for reproducibility.
      </p>
    </article>
  );
}
