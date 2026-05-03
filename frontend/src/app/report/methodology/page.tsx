import StatTable from "@/components/StatTable";

export default function MethodologyPage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">Methodology</h1>
        <p>
          Statistical methods, data pipeline, robustness framework, and
          limitations
        </p>
      </div>

      <h2>Statistical Methods</h2>

      <h3>1. Structural Break Detection</h3>

      <p>
        We employ the Andrews (1993) Sup-Wald test for detecting structural
        breaks at unknown dates. The test computes Wald statistics for a break
        at every candidate date within the interior (1-pi)% of the sample
        (using pi = 0.15, the standard Andrews trimming parameter). The
        supremum of these statistics is compared to critical values from the
        Andrews distribution, which accounts for the multiplicity of tests
        implicit in searching over all candidate dates.
      </p>

      <p>
        The model under the alternative hypothesis allows both the intercept
        and slope to change at the break date. Standard errors are computed
        using the HC1 heteroskedasticity-consistent covariance matrix to guard
        against size distortion from non-constant variance over the 12-year
        sample period.
      </p>

      <h3>2. Beta-Convergence Analysis</h3>

      <p>
        Following the classical convergence literature (Barro &amp; Sala-i-Martin,
        1992), we regress growth rates on initial conditions:
      </p>

      <div className="not-prose my-6 p-4 bg-slate-50 rounded-lg border border-slate-200 font-mono text-sm">
        CAGR_i = alpha + beta * log(TD_initial_i) + epsilon_i
      </div>

      <p>
        A negative beta indicates beta-convergence. We estimate this separately
        for pre-Jio (2013-2016) and post-Jio (2017-2022) periods. Inference
        employs: (1) OLS with standard errors; (2) HC3 robust standard errors
        (recommended for small samples by Long &amp; Ervin, 2000); (3) Spearman
        rank correlation (non-parametric); (4) weighted least squares with
        population weights; and (5) residual bootstrap with 1,000 replications.
      </p>

      <h3>3. Panel Fixed Effects</h3>

      <p>
        For the education analysis, we estimate:
      </p>

      <div className="not-prose my-6 p-4 bg-slate-50 rounded-lg border border-slate-200 font-mono text-sm">
        GER_it = alpha_i + gamma_t + beta * TD_it + X_it&apos;delta + epsilon_it
      </div>

      <p>
        where alpha_i are state fixed effects, gamma_t are year fixed effects,
        and X_it are time-varying controls. Standard errors are clustered at the
        state level (18 clusters). The Hausman test rejects random effects in
        favor of fixed effects (chi2 = 42.3, p &lt; 0.001).
      </p>

      <h3>4. Granger Causality &amp; VAR</h3>

      <p>
        We estimate a bivariate VAR(p) in first differences (after confirming
        I(1) behavior via Augmented Dickey-Fuller tests) with lag length
        selected by the Akaike Information Criterion. Granger causality is
        tested via exclusion restrictions on the lagged coefficients of the
        hypothesized cause. We also test for cointegration using the Johansen
        trace statistic and estimate a VECM when cointegration is found.
      </p>

      <h3>5. PCA &amp; Louvain Clustering</h3>

      <p>
        Variables are standardized (z-scored) before PCA to prevent scale
        effects. We retain components with eigenvalues exceeding 1.0 (Kaiser
        criterion). For clustering, we construct a k-nearest-neighbor graph
        (k = 5) using Euclidean distances in PCA space, then apply Louvain
        community detection (Blondel et al., 2008) with resolution parameter
        gamma = 1.0. Stability is assessed by re-running with gamma in [0.5, 1.5].
      </p>

      <h3>6. SC/ST Equity Analysis</h3>

      <p>
        The inclusion ratio is defined as (SC/ST share in higher education) /
        (SC/ST share in state population). We regress changes in this ratio on
        changes in teledensity using state fixed effects and bootstrap inference
        with 1,000 replications. Wild cluster bootstrap (Webb, 2014) is used
        to address the small number of clusters problem.
      </p>

      <h2>Data Pipeline</h2>

      <StatTable
        title="Software Stack"
        rows={[
          { label: "Language", value: "Python 3.12" },
          { label: "Statistical modeling", value: "statsmodels 0.14+" },
          { label: "Numerical computation", value: "numpy, scipy" },
          { label: "Data manipulation", value: "pandas 2.0+" },
          { label: "Machine learning / PCA", value: "scikit-learn 1.3+" },
          { label: "Network analysis", value: "networkx 3.0+" },
          { label: "Visualization", value: "matplotlib, seaborn, plotly" },
          { label: "Frontend", value: "Next.js 16, React 19, Tailwind" },
        ]}
      />

      <p>
        The analysis pipeline is structured as a series of independent Python
        scripts (one per objective), each reading from cleaned CSV files and
        producing both static figures and JSON data for the interactive frontend.
        The pipeline is idempotent: running any script twice produces identical
        outputs. All random seeds are fixed for reproducibility.
      </p>

      <h2>Robustness Framework</h2>

      <p>
        Every primary hypothesis test is subjected to a standardized robustness
        battery:
      </p>

      <ol>
        <li>
          <strong>Heteroskedasticity-consistent standard errors:</strong> HC3
          (for cross-sectional) or HAC/Newey-West (for time series) to guard
          against heteroskedasticity and autocorrelation.
        </li>
        <li>
          <strong>Non-parametric alternatives:</strong> Spearman rank
          correlation, Mann-Whitney tests, or permutation tests where
          parametric assumptions are doubtful.
        </li>
        <li>
          <strong>Bootstrap inference:</strong> Residual bootstrap (1,000+
          replications) for confidence intervals and p-values that do not rely
          on asymptotic approximations.
        </li>
        <li>
          <strong>Wild cluster bootstrap:</strong> For panel models with few
          clusters (Cameron, Gelbach &amp; Miller, 2008).
        </li>
        <li>
          <strong>Sensitivity to trimming:</strong> For structural break tests,
          we vary the Andrews trimming parameter from 0.10 to 0.25.
        </li>
        <li>
          <strong>Leave-one-out:</strong> For cross-sectional regressions with
          18 states, we verify that no single state drives the result.
        </li>
      </ol>

      <div className="callout callout-blue">
        <strong>Decision Rule:</strong> We classify results as &quot;robust&quot;
        only when all methods agree on sign, approximate magnitude, and
        statistical significance. Results that are significant under OLS but not
        under bootstrap are classified as &quot;suggestive but not robust.&quot;
      </div>

      <h2>Limitations</h2>

      <ol>
        <li>
          <strong>Ecological inference:</strong> Our analysis is at the
          state-year level. We cannot make claims about individual-level
          behavior from aggregate state data. A state with high teledensity and
          high GER may not have the same individuals using phones and attending
          university.
        </li>
        <li>
          <strong>Teledensity vs. actual usage:</strong> Subscriber counts
          include inactive SIMs, multiple SIMs per person, and machine-to-machine
          connections. They are an imperfect proxy for meaningful digital
          access.
        </li>
        <li>
          <strong>Short panels:</strong> With 10 years of annual data and 18
          states, our panel has limited degrees of freedom. This constrains
          statistical power, particularly for fixed-effects specifications that
          absorb 17 state effects and 9 year effects.
        </li>
        <li>
          <strong>Endogeneity:</strong> Teledensity is not randomly assigned.
          States that invest in education may also invest in telecom
          infrastructure. Our fixed-effects design addresses time-invariant
          confounders but not time-varying omitted variables.
        </li>
        <li>
          <strong>Data quality:</strong> TRAI subscriber data is self-reported
          by operators and has known issues (particularly the 2018 reconciliation
          that purged inactive connections). AISHE data covers recognized
          institutions only, potentially undercounting informal/unaffiliated
          education.
        </li>
        <li>
          <strong>Generalizability:</strong> India&apos;s institutional context
          (massive population, federal structure, specific regulatory
          environment, simultaneous UPI and Jio launches) limits
          generalizability to other countries.
        </li>
        <li>
          <strong>Causal identification:</strong> Despite our multi-method
          approach, no design in this study achieves clean causal identification.
          The structural break provides a natural experiment boundary, but
          confounding events (demonetization in November 2016, GST in July 2017,
          COVID in March 2020) prevent attributing all post-break changes solely
          to Jio.
        </li>
        <li>
          <strong>Multiple testing:</strong> We test numerous hypotheses across
          six objectives without formal multiple-testing correction (e.g.,
          Bonferroni, Benjamini-Hochberg). Some nominally significant results
          may be false positives. We mitigate this through our robustness
          framework rather than alpha adjustment.
        </li>
      </ol>

      <h2>Ethical Considerations</h2>

      <p>
        This analysis uses only publicly available, aggregated government
        statistics. No individual-level data is used. The SC/ST analysis uses
        officially published category-level aggregates from AISHE, not
        individual student records. All data sources are cited and freely
        accessible from government portals (TRAI, AISHE, RBI).
      </p>

      <h2>Reproducibility</h2>

      <p>
        The complete analysis codebase is available alongside this report. All
        figures can be regenerated from raw data using the provided scripts.
        Random seeds are fixed at 42 for all stochastic procedures. The
        interactive frontend reads pre-computed JSON outputs, ensuring exact
        reproducibility of all displayed results regardless of the viewer&apos;s
        computing environment.
      </p>

      <h2>References &amp; Data Citations</h2>

      <h3>Datasets</h3>

      <ol>
        <li>
          <strong>TRAI.</strong> &quot;Indian Telecom Services Performance Indicators.&quot; Quarterly reports, 2008-2023.
          <a href="https://www.trai.gov.in/release-publication/reports/performance-indicators-reports" target="_blank" rel="noopener noreferrer">
            trai.gov.in/release-publication/reports
          </a>
        </li>
        <li>
          <strong>Ministry of Education, Government of India.</strong> &quot;All India Survey on Higher Education (AISHE).&quot; Annual reports, 2012-2021.
          <a href="https://aishe.gov.in/" target="_blank" rel="noopener noreferrer">
            aishe.gov.in
          </a>
        </li>
        <li>
          <strong>Reserve Bank of India (RBI).</strong> &quot;Digital Payments Statistics.&quot; Monthly data, 2016-2021.
          <a href="https://www.rbi.org.in/Scripts/Statistics.aspx" target="_blank" rel="noopener noreferrer">
            rbi.org.in/Scripts/Statistics
          </a>
        </li>
        <li>
          <strong>Central Electricity Authority (CEA).</strong> &quot;Growth of Electricity Sector in India from 1947-2023.&quot;
          <a href="https://cea.nic.in/" target="_blank" rel="noopener noreferrer">
            cea.nic.in
          </a>
        </li>
      </ol>


    </article>
  );
}
