"use client";

import Figure from "@/components/Figure";
import StatTable from "@/components/StatTable";
import InteractiveChart from "@/components/InteractiveChart";

export default function ConvergencePage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">Convergence &amp; Stasis</h1>
        <p>
          The strongest finding: states were converging before Jio, then stopped
        </p>
      </div>

      <div className="callout callout-green">
        <strong>Headline Finding:</strong> Before Jio, Indian states exhibited
        statistically significant beta-convergence in teledensity (p = 0.003,
        HC3 robust). After Jio&apos;s entry, this convergence vanishes entirely
        (p = 0.667). States that were behind stayed behind.
      </div>

      <h2>The Convergence Hypothesis</h2>

      <p>
        Beta-convergence is a foundational concept in growth economics: it asks
        whether entities that start behind grow faster, thereby &quot;catching
        up&quot; to leaders over time. In the context of Indian
        telecommunications, we ask: were states with lower initial teledensity
        (subscribers per 100 population) growing their networks faster than
        states that were already well-connected?
      </p>

      <p>
        This is not merely an academic question. If convergence holds, the
        digital divide is self-correcting: lagging states naturally catch up
        through market forces. If convergence fails, active policy intervention
        becomes necessary to prevent the gap from widening or ossifying. Our
        analysis reveals a dramatic shift: convergence was operating before
        2016, then abruptly ceased.
      </p>

      <h2>Pre-Jio Period: Clear Convergence</h2>

      <p>
        For the pre-Jio period (2013-2016), we regress the compound annual
        growth rate (CAGR) of state teledensity on initial teledensity levels.
        Under beta-convergence, this coefficient should be negative: states
        starting at lower levels grow faster.
      </p>

      <InteractiveChart
        dataFile="convergence_pre_jio.json"
        title="Interactive: Pre-Jio Beta-Convergence (2013-2016)"
        chartConfig={(data) => {
          const x = data.map((d) => d.initial as number);
          const y = data.map((d) => d.cagr as number);
          const labels = data.map((d) => d.state as string);

          // Simple linear regression for trend line
          const n = x.length;
          const sumX = x.reduce((a, b) => a + b, 0);
          const sumY = y.reduce((a, b) => a + b, 0);
          const sumXY = x.reduce((a, xi, i) => a + xi * y[i], 0);
          const sumX2 = x.reduce((a, xi) => a + xi * xi, 0);
          const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
          const intercept = (sumY - slope * sumX) / n;

          const xMin = Math.min(...x);
          const xMax = Math.max(...x);

          return {
            data: [
              {
                x,
                y,
                type: "scatter",
                mode: "markers+text",
                name: "States",
                text: labels,
                textposition: "top center",
                textfont: { size: 9, color: "#6b7280" },
                marker: {
                  size: 12,
                  color: x.map((v) => v),
                  colorscale: "Viridis",
                  showscale: true,
                  colorbar: { title: "Initial TD", thickness: 15 },
                  line: { width: 1, color: "white" },
                },
              },
              {
                x: [xMin, xMax],
                y: [intercept + slope * xMin, intercept + slope * xMax],
                type: "scatter",
                mode: "lines",
                name: `Trend (slope=${slope.toFixed(3)})`,
                line: { color: "#ef4444", width: 2.5, dash: "dash" },
              },
            ],
            layout: {
              title: {
                text: "Beta-Convergence: Initial Teledensity vs. Growth Rate",
                font: { size: 14 },
              },
              xaxis: { title: "Initial Teledensity (2013)" },
              yaxis: { title: "CAGR 2013-2016 (%)" },
              showlegend: true,
              legend: { x: 0.7, y: 0.98 },
            },
          };
        }}
      />

      <p>
        The scatter plot reveals a clear negative relationship: states like
        Jammu &amp; Kashmir (initial TD: 66.8, CAGR: 14.0%) and Himachal Pradesh
        (initial TD: 105.6, CAGR: 12.9%) grew rapidly from low bases, while
        Delhi (initial TD: 226.8, CAGR: 4.5%) and Punjab (initial TD: 107.2,
        CAGR: 4.3%) grew slowly from high bases. This is textbook
        beta-convergence.
      </p>

      <h2>Robustness: Multiple Inference Methods</h2>

      <p>
        A single OLS regression could be misleading due to outliers, leverage
        points, or heteroskedasticity. We therefore test the convergence
        relationship using four distinct methods, each addressing a different
        potential threat to inference:
      </p>

      <StatTable
        title="Pre-Jio Convergence: Robustness Battery"
        rows={[
          { label: "OLS (standard errors)", value: "beta = -0.044, p = 0.030", significant: true },
          { label: "OLS (HC3 robust SE)", value: "beta = -0.044, p = 0.003", significant: true },
          { label: "Spearman Rank Correlation", value: "rho = -0.56, p = 0.015", significant: true },
          { label: "Weighted Least Squares", value: "beta = -0.039, p = 0.022", significant: true },
          { label: "Bootstrap (1000 reps)", value: "95% CI: [-0.071, -0.012]", significant: true },
        ]}
      />

      <div className="callout callout-blue">
        <strong>All five methods agree:</strong> Before Jio, lower-teledensity
        states grew significantly faster. The HC3 result (p = 0.003) is actually
        stronger than OLS, suggesting the heteroskedasticity works in our favor.
        The Spearman test confirms the relationship is robust to outliers and
        non-linearity.
      </div>

      <h2>Post-Jio Period: Convergence Vanishes</h2>

      <p>
        We repeat the identical analysis for the post-Jio period (2017-2022).
        The results are strikingly different:
      </p>

      <StatTable
        title="Post-Jio Period: Convergence Tests"
        rows={[
          { label: "OLS (HC3 robust SE)", value: "beta = -0.008, p = 0.667" },
          { label: "Spearman Rank Correlation", value: "rho = -0.12, p = 0.634" },
          { label: "Bootstrap (1000 reps)", value: "95% CI: [-0.042, +0.029]" },
          { label: "Effect size (Cohen's d)", value: "0.11 (negligible)" },
        ]}
      />

      <p>
        The convergence coefficient collapses from -0.044 to -0.008 and loses
        all statistical significance. The confidence interval straddles zero
        comfortably. States that were behind in 2017 did not systematically grow
        faster than states that were ahead. The catch-up process halted.
      </p>

      <Figure
        src="/figures/obj6_convergence.png"
        alt="Convergence comparison pre and post Jio"
        caption="Figure 3.1: Beta-convergence scatter plots for pre-Jio (2013-2016) and post-Jio (2017-2022) periods. The negative slope vanishes in the post-Jio era."
      />

      <h2>Why Did Convergence Stop?</h2>

      <p>
        This is the central puzzle of our analysis. We propose three
        complementary mechanisms:
      </p>

      <ol>
        <li>
          <strong>Universal Shock, Uniform Impact:</strong> Jio offered the same
          free/cheap data everywhere simultaneously. Unlike the pre-Jio era,
          where lagging states had more &quot;room to grow,&quot; Jio&apos;s
          blanket disruption lifted all states roughly equally, preserving
          existing gaps.
        </li>
        <li>
          <strong>Infrastructure Constraints Bind:</strong> After the
          &quot;easy&quot; urban subscribers were acquired, further growth
          requires physical tower infrastructure in rural areas. States with
          lower initial teledensity often lack this infrastructure, limiting
          their ability to absorb new subscribers regardless of price.
        </li>
        <li>
          <strong>Demand Saturation at the Top:</strong> High-teledensity states
          like Delhi (TD &gt; 250) were already saturated. Their slow pre-Jio
          growth was not inefficiency but simply hitting a ceiling. Post-Jio,
          even lagging states began approaching similar ceilings faster.
        </li>
      </ol>

      <h2>Implications for Digital Equity</h2>

      <p>
        The cessation of convergence has profound policy implications. If the
        digital divide in India is no longer self-correcting through market
        forces, then targeted interventions&mdash;such as the BharatNet
        programme for rural fiber connectivity&mdash;become not merely
        complementary to market provision but essential for preventing permanent
        stratification. Our analysis in the &quot;Who&apos;s Left Behind&quot;
        chapter quantifies exactly how large these gaps remain.
      </p>

      <div className="callout callout-amber">
        <strong>Causal Interpretation Caveat:</strong> While the timing is
        suggestive, we cannot claim that Jio caused the cessation of
        convergence. The structural break merely divides time into two periods.
        Other contemporaneous changes (demonetization, GST rollout, COVID-19)
        could contribute to the post-2016 pattern. We present this as a robust
        correlation, not a causal claim.
      </div>

      <Figure
        src="/figures/obj6_leapfrogging.png"
        alt="Leapfrogging analysis"
        caption="Figure 3.2: Analysis of potential leapfrogging patterns across states, examining whether any lagging states managed to overtake leaders."
      />

      <Figure
        src="/figures/obj6_competition_adoption.png"
        alt="Competition and adoption dynamics"
        caption="Figure 3.3: Relationship between market competition (HHI) and adoption patterns across states."
      />
    </article>
  );
}
