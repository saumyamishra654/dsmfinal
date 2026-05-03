"use client";

import Figure from "@/components/Figure";
import StatTable from "@/components/StatTable";
import InteractiveChart from "@/components/InteractiveChart";

export default function ShockPage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">The Exogenous Shock</h1>
        <p>
          Formally identifying Jio&apos;s entry as a structural break in Indian
          telecommunications
        </p>
      </div>

      <h2>Background: A Market Disruption Unlike Any Other</h2>

      <p>
        On September 5, 2016, Reliance Jio Infocomm Limited launched commercial
        operations with an offer that sent shockwaves through the global
        telecommunications industry: unlimited 4G data, free of charge, for all
        new subscribers. The &quot;Jio Welcome Offer&quot; was not a marginal
        price reduction or a targeted promotion&mdash;it was a wholesale
        elimination of the price barrier to mobile internet access for over 1.3
        billion people.
      </p>

      <p>
        The scale of the resulting disruption is difficult to overstate. Within
        six months of launch, Jio had acquired over 100 million subscribers,
        making it the fastest-growing mobile operator in history. The incumbent
        operators&mdash;Airtel, Vodafone, Idea&mdash;were forced into a brutal
        price war that would see data costs fall by over 95% within two years.
        Average monthly data consumption per user jumped from under 1 GB to over
        11 GB, a figure that exceeded even many developed nations.
      </p>

      <p>
        But was this truly a <em>structural break</em> in the statistical sense?
        Or was India&apos;s subscriber growth already accelerating, with Jio
        merely riding an existing wave? To answer this, we deploy the Andrews
        (1993) Sup-Wald test for unknown breakpoints&mdash;a method that does
        not require the researcher to specify when the break occurred, instead
        searching endogenously for the most likely breakpoint.
      </p>

      <h2>The Structural Break Test</h2>

      <p>
        We model national wireless subscribers as a function of a linear time
        trend, testing for a break in both the intercept and slope at an unknown
        date within the interior 70% of the sample (the Andrews trimming
        parameter). The null hypothesis is parameter constancy&mdash;no
        structural break at any point in the series.
      </p>

      <StatTable
        title="Sup-Wald Structural Break Results"
        rows={[
          { label: "Sup-F Statistic", value: "253.60", significant: true },
          { label: "Critical Value (5%)", value: "12.35" },
          { label: "Identified Break Date", value: "September 2016", significant: true },
          { label: "Pre-break Trend (monthly)", value: "+4.2M subscribers" },
          { label: "Post-break Trend (monthly)", value: "+12.8M subscribers" },
          { label: "Ratio (Post/Pre slope)", value: "3.05x", significant: true },
        ]}
      />

      <div className="callout callout-green">
        <strong>Result:</strong> The Sup-F statistic of 253.60 exceeds the 5%
        critical value of 12.35 by a factor of twenty. This is among the most
        decisive structural break results in the applied economics literature on
        telecommunications. The identified break date aligns precisely with
        Jio&apos;s commercial launch in September 2016.
      </div>

      <p>
        The test identifies September 2016 as the most likely breakpoint with
        overwhelming statistical confidence. Before Jio, India was adding
        approximately 4.2 million wireless subscribers per month on a steady
        linear trend. After the break, this rate tripled to 12.8 million per
        month in the initial expansion phase. The post-break trajectory then
        stabilized at a higher plateau of roughly 1.1 billion total subscribers.
      </p>

      <h2>National Wireless Subscriber Growth</h2>

      <InteractiveChart
        dataFile="national_wireless.json"
        title="Interactive: National Wireless Subscribers (2009-2021)"
        chartConfig={(data) => {
          const years = data.map(
            (d) => `${d.year}-${String(d.month).padStart(2, "0")}-01`
          );
          const values = data.map((d) => d.total_millions as number);
          return {
            data: [
              {
                x: years,
                y: values,
                type: "scatter",
                mode: "lines",
                name: "Total Subscribers",
                line: { color: "#3b82f6", width: 2.5 },
                fill: "tozeroy",
                fillcolor: "rgba(59, 130, 246, 0.08)",
              },
              {
                x: ["2016-09-01", "2016-09-01"],
                y: [0, Math.max(...values)],
                type: "scatter",
                mode: "lines",
                name: "Jio Launch (Sep 2016)",
                line: { color: "#ef4444", width: 2, dash: "dash" },
              },
            ],
            layout: {
              title: {
                text: "India Wireless Subscribers (Millions)",
                font: { size: 14 },
              },
              xaxis: { title: "Date", type: "date" },
              yaxis: { title: "Subscribers (Millions)" },
              showlegend: true,
              legend: { x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.8)" },
              annotations: [
                {
                  x: "2016-09-01",
                  y: 1000,
                  xref: "x",
                  yref: "y",
                  text: "Jio Launch",
                  showarrow: true,
                  arrowhead: 2,
                  ax: -60,
                  ay: -40,
                  font: { color: "#ef4444", size: 11, family: "Inter" },
                },
              ],
            },
          };
        }}
      />

      <p>
        The interactive chart above allows you to zoom into any period and hover
        over data points for exact values. Notice the distinctive
        &quot;knee&quot; in the curve around September 2016, where the growth
        trajectory shifts abruptly upward. Also notable is the brief plateau
        around 2012, when the 2G spectrum allocation controversy temporarily
        depressed subscriber growth.
      </p>

      <h2>Market Structure Transformation</h2>

      <p>
        Jio&apos;s entry did not merely add subscribers&mdash;it fundamentally
        restructured the competitive landscape. The Herfindahl-Hirschman Index
        (HHI), a standard measure of market concentration, fell from
        approximately 2,500 (indicating a moderately concentrated market) to
        roughly 1,800 (indicating a competitive market) within two years of
        Jio&apos;s launch.
      </p>

      <Figure
        src="/figures/obj1_hhi_over_time.png"
        alt="HHI market concentration over time"
        caption="Figure 2.1: Herfindahl-Hirschman Index showing market deconcentration following Jio's entry. The drop from ~2500 to ~1800 indicates a transition from moderate concentration to a competitive market structure."
      />

      <p>
        This structural shift was accompanied by dramatic consolidation among
        incumbents. The number of major operators fell from over a dozen to
        effectively four (Jio, Airtel, Vodafone-Idea, BSNL). Paradoxically, the
        HHI decreased even as the number of players fell, because Jio captured
        market share from the combined tail of smaller operators, creating a
        more evenly distributed oligopoly.
      </p>

      <Figure
        src="/figures/obj1_provider_market_share.png"
        alt="Provider market share evolution"
        caption="Figure 2.2: Evolution of provider market shares showing Jio's rapid rise and the consolidation of smaller operators."
      />

      <h2>The Growth Trajectory</h2>

      <Figure
        src="/figures/obj1_national_wireless.png"
        alt="National wireless subscriber time series"
        caption="Figure 2.3: The complete national wireless subscriber series with the identified structural break overlaid."
      />

      <p>
        Three distinct phases are visible in the subscriber trajectory: (1) the
        rapid expansion phase from 2009-2012, driven by falling handset costs
        and regulatory liberalization; (2) the plateau phase from 2012-2016,
        where growth slowed as the &quot;easy&quot; urban market saturated; and
        (3) the Jio-accelerated phase from 2016 onward, where cheap data drove
        rural and secondary-device adoption.
      </p>

      <h2>Implications</h2>

      <p>
        The formal identification of a structural break matters for several
        reasons. First, it validates using September 2016 as a natural
        experiment boundary in subsequent analyses&mdash;we can legitimately
        compare &quot;pre-Jio&quot; and &quot;post-Jio&quot; periods with
        statistical backing. Second, the magnitude of the break (Sup-F exceeding
        its critical value by 20x) suggests this was not a gradual transition
        but a genuine regime change in India&apos;s telecommunications sector.
      </p>

      <p>
        The remaining chapters of this report investigate what this regime
        change meant for convergence across states, educational access, digital
        financial inclusion, and the persistent divide between India&apos;s
        digital leaders and laggards.
      </p>

      <div className="callout callout-amber">
        <strong>Methodological Note:</strong> The Sup-Wald test is preferred
        over the Chow test here because it does not require pre-specifying the
        break date. The critical values are from Andrews (1993, Table 1) for 2
        restrictions. Heteroskedasticity-consistent covariance matrices are used
        throughout.
      </div>
    </article>
  );
}
