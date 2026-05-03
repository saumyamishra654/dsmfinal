"use client";

import Figure from "@/components/Figure";
import StatTable from "@/components/StatTable";
import InteractiveChart from "@/components/InteractiveChart";

export default function EducationPage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">Connectivity &amp; Education</h1>
        <p>
          Exploring the relationship between teledensity and higher education
          enrolment across Indian states
        </p>
      </div>

      <h2>The Cross-Sectional Story: A Strong Correlation</h2>

      <p>
        Across Indian states, the correlation between teledensity (wireless
        subscribers per 100 population) and Gross Enrolment Ratio (GER) in
        higher education is striking. With a cross-sectional correlation
        coefficient of approximately <strong>r = 0.77</strong>, states with
        higher connectivity consistently show higher educational participation.
        This is one of the strongest bivariate relationships in our dataset.
      </p>

      <p>
        The relationship is intuitive: mobile internet enables access to
        educational resources, online courses, distance learning programmes, and
        information about educational opportunities. Students in connected
        states can research universities, complete online applications, and
        access supplementary learning materials. Faculty can adopt blended
        learning approaches that increase institutional capacity without
        proportional infrastructure investment.
      </p>

      <Figure
        src="/figures/obj2_scatter.png"
        alt="Scatter plot of teledensity vs GER"
        caption="Figure 4.1: Cross-sectional relationship between teledensity and Gross Enrolment Ratio. Each point represents a state-year observation. The correlation (r ~ 0.77) is among the strongest in our analysis."
      />

      <h2>Temporal Evolution: Strengthening Over Time</h2>

      <p>
        The correlation is not static. Between 2013 and 2022, the
        teledensity-GER relationship strengthened over time, suggesting that as
        connectivity penetrates deeper into educational ecosystems, its
        relationship with enrolment becomes more pronounced. This is consistent
        with network effects: the value of connectivity for education increases
        as more educational resources move online and as peer effects (seeing
        friends use phones for learning) compound.
      </p>

      <Figure
        src="/figures/obj2_correlation_over_time.png"
        alt="Correlation between teledensity and GER over time"
        caption="Figure 4.2: Evolution of the Pearson correlation coefficient between state teledensity and GER across years. The strengthening trend suggests deepening institutional reliance on connectivity."
      />

      <h2>Panel Regression: The Within-State Challenge</h2>

      <p>
        Cross-sectional correlations are suggestive but not causal. Rich, urban,
        well-governed states naturally have both more cell towers and more
        universities. The critical test is whether <em>changes</em> in
        teledensity within a state predict <em>changes</em> in educational
        outcomes, holding constant all time-invariant state characteristics.
      </p>

      <p>
        We estimate a two-way fixed effects model with state and year fixed
        effects, clustering standard errors by state:
      </p>

      <StatTable
        title="Panel Fixed Effects: Teledensity on Total GER"
        rows={[
          { label: "Coefficient (beta)", value: "0.082" },
          { label: "Clustered SE", value: "0.054" },
          { label: "t-statistic", value: "1.52" },
          { label: "p-value", value: "0.14" },
          { label: "95% CI", value: "[-0.028, +0.192]" },
          { label: "R-squared (within)", value: "0.31" },
          { label: "N (state-years)", value: "180" },
        ]}
      />

      <div className="callout callout-amber">
        <strong>Honest Assessment:</strong> The panel fixed-effects estimate is
        positive (beta = 0.082) and economically meaningful&mdash;a 10-point
        increase in teledensity is associated with a 0.82 percentage point
        increase in GER. However, the estimate is not statistically significant
        at conventional levels (p = 0.14). We cannot reject the null hypothesis
        of no within-state causal effect.
      </div>

      <p>
        This null result does not mean connectivity has no effect on education.
        It means that with 18 states observed over 10 years, and with the
        substantial noise in both series, we lack the statistical power to
        detect what may be a real but modest effect. The confidence interval
        [-0.028, +0.192] includes both zero and economically meaningful positive
        values.
      </p>

      <h2>State-Level Teledensity Trajectories</h2>

      <InteractiveChart
        dataFile="tele_density_panel.json"
        title="Interactive: State Teledensity Over Time (2013-2022)"
        chartConfig={(data) => {
          const states = [...new Set(data.map((d) => d.state as string))];
          const colors = [
            "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
            "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
            "#14b8a6", "#e11d48", "#0ea5e9", "#a855f7", "#d946ef",
            "#64748b", "#059669", "#dc2626",
          ];

          const traces = states.map((state, i) => {
            const stateData = data.filter((d) => d.state === state);
            return {
              x: stateData.map((d) => d.year as number),
              y: stateData.map((d) => d.td as number),
              type: "scatter",
              mode: "lines+markers",
              name: state,
              line: { color: colors[i % colors.length], width: 2 },
              marker: { size: 4 },
            };
          });

          return {
            data: traces,
            layout: {
              title: {
                text: "Teledensity by State (Subscribers per 100 Population)",
                font: { size: 14 },
              },
              xaxis: { title: "Year", dtick: 1 },
              yaxis: { title: "Teledensity" },
              showlegend: true,
              legend: {
                x: 1.02,
                y: 1,
                font: { size: 9 },
              },
              margin: { r: 150 },
            },
          };
        }}
        height={600}
      />

      <p>
        The interactive chart reveals considerable heterogeneity in teledensity
        trajectories. Delhi dominates at over 250 subscribers per 100 people
        (reflecting multiple SIM ownership). Most states cluster between 60 and
        130, with Bihar conspicuously lagging at around 55. Note the common
        peak-and-decline pattern post-2016, where initially inflated subscriber
        counts (from free Jio SIMs) normalized as inactive connections were
        purged.
      </p>

      <h2>Regression Coefficients Across Specifications</h2>

      <Figure
        src="/figures/obj2_regression_coefficients.png"
        alt="Regression coefficients across model specifications"
        caption="Figure 4.3: Coefficient estimates and confidence intervals across different model specifications. The fixed-effects estimate is positive but wide."
      />

      <p>
        The figure above shows how the teledensity coefficient varies across
        model specifications. The pooled OLS estimate (which conflates
        cross-sectional and within-state variation) is large and significant.
        The fixed-effects estimate (which isolates within-state variation) is
        smaller and insignificant. This pattern is consistent with
        omitted-variable bias in the cross-sectional estimates: unobserved state
        characteristics (urbanization, income, governance quality) drive both
        teledensity and GER.
      </p>

      <h2>Key Takeaways</h2>

      <ol>
        <li>
          States with higher connectivity consistently have higher educational
          participation (cross-sectional r = 0.77).
        </li>
        <li>
          This relationship strengthens over time, suggesting deepening
          complementarity between digital access and educational systems.
        </li>
        <li>
          Within-state panel evidence is directionally positive (beta = 0.082)
          but not statistically robust&mdash;we cannot establish causality with
          available data.
        </li>
        <li>
          The disconnect between cross-sectional and panel results highlights
          the importance of confounders: state-level development drives both
          outcomes simultaneously.
        </li>
      </ol>

      <div className="callout callout-blue">
        <strong>Policy Implication:</strong> While we cannot prove that
        increasing teledensity causes higher enrolment, the strong and
        strengthening correlation suggests connectivity is at minimum a
        complementary input to educational infrastructure. Policies that expand
        connectivity in educationally lagging states are unlikely to harm and
        may meaningfully support educational access.
      </div>
    </article>
  );
}
