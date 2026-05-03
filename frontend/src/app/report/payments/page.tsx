"use client";

import Figure from "@/components/Figure";
import StatTable from "@/components/StatTable";
import InteractiveChart from "@/components/InteractiveChart";

export default function PaymentsPage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">Digital Payments Revolution</h1>
        <p>
          How UPI transformed India&apos;s financial landscape and whether
          mobile connectivity drives digital payment adoption
        </p>
      </div>

      <h2>The UPI Phenomenon</h2>

      <p>
        India&apos;s Unified Payments Interface (UPI) represents arguably the
        most successful public digital infrastructure project in history. Launched
        in August 2016&mdash;nearly simultaneously with Jio&mdash;UPI created a
        free, interoperable, real-time payment rail that any bank or fintech
        company could build upon. The synergy between cheap mobile internet (via
        Jio) and free digital payments (via UPI) created a flywheel effect of
        extraordinary magnitude.
      </p>

      <p>
        Between April 2017 and July 2022, UPI transaction volume grew from
        0.7 crore (7 million) transactions per month to over 586 crore (5.86
        billion)&mdash;an 83,700% increase in just five years. This growth
        simultaneously cannibalized traditional debit card transactions, which
        stagnated at around 30-45 crore per month while UPI volumes
        skyrocketed.
      </p>

      <InteractiveChart
        dataFile="digital_transactions.json"
        title="Interactive: Digital Payments Evolution (2017-2022)"
        chartConfig={(data) => {
          const dates = data.map((d) => d.date as string);
          const digital = data.map((d) => d.digital_txn_crores as number);
          const bhim = data.map((d) => d.bhim_txn_crores as number);
          const debit = data.map((d) => d.debit_card_crores as number);

          return {
            data: [
              {
                x: dates,
                y: digital,
                type: "scatter",
                mode: "lines",
                name: "Total Digital Transactions",
                line: { color: "#3b82f6", width: 2.5 },
                fill: "tozeroy",
                fillcolor: "rgba(59, 130, 246, 0.05)",
              },
              {
                x: dates,
                y: bhim,
                type: "scatter",
                mode: "lines",
                name: "UPI/BHIM Transactions",
                line: { color: "#10b981", width: 2.5 },
                fill: "tozeroy",
                fillcolor: "rgba(16, 185, 129, 0.05)",
              },
              {
                x: dates,
                y: debit,
                type: "scatter",
                mode: "lines",
                name: "Debit Card Transactions",
                line: { color: "#f59e0b", width: 2 },
              },
            ],
            layout: {
              title: {
                text: "Monthly Digital Payment Volumes (Crore Transactions)",
                font: { size: 14 },
              },
              xaxis: { title: "Date", type: "date" },
              yaxis: { title: "Transactions (Crores)" },
              showlegend: true,
              legend: { x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.9)" },
              annotations: [
                {
                  x: "2020-04-01",
                  y: 303,
                  xref: "x",
                  yref: "y",
                  text: "COVID Lockdown",
                  showarrow: true,
                  arrowhead: 2,
                  ax: 40,
                  ay: -40,
                  font: { size: 10 },
                },
              ],
            },
          };
        }}
        height={500}
      />

      <h2>Payment Mode Shares: A Structural Shift</h2>

      <p>
        The interactive chart above tells a story of complete modal
        transformation. In early 2017, UPI constituted less than 1% of digital
        transactions. By mid-2022, it accounted for over 60% of all digital
        payment volume. Debit cards, which once dominated retail digital
        payments, now represent less than 5% of monthly transaction volume.
      </p>

      <Figure
        src="/figures/obj3_payment_shares.png"
        alt="Payment mode share evolution"
        caption="Figure 6.1: Evolution of payment mode shares, showing UPI's complete dominance of the digital payments landscape by 2022."
      />

      <p>
        This is not merely a substitution effect (replacing cards with phones).
        The total volume of digital transactions grew from ~160 crore per month
        to over 1,000 crore per month. UPI unlocked entirely new use
        cases&mdash;street vendors, auto-rickshaw drivers, chai stalls&mdash;that
        were never served by cards. The growth is predominantly at the extensive
        margin (new users and new use cases) rather than the intensive margin
        (existing card users switching to phones).
      </p>

      <h2>Seasonal Decomposition: Underlying Trend</h2>

      <Figure
        src="/figures/obj3_stl_decomposition.png"
        alt="STL decomposition of digital transactions"
        caption="Figure 6.2: Seasonal-Trend-Loess decomposition of monthly digital transaction volumes. The trend component shows near-exponential growth with a brief COVID dip."
      />

      <p>
        The STL decomposition reveals important structure in the data. The
        trend component is nearly exponential through 2019, suffers a sharp
        COVID-induced dip in April 2020, then resumes even steeper growth.
        The seasonal component shows March spikes (fiscal year-end) and
        October-November peaks (festive season purchases). The residual is
        well-behaved, suggesting the trend-season model captures the data
        generating process well.
      </p>

      <h2>Granger Causality: Does Connectivity Drive Payments?</h2>

      <p>
        We test whether mobile subscriber growth Granger-causes digital
        transaction growth (and vice versa) using a bivariate VAR framework
        with optimal lag length selected by AIC.
      </p>

      <StatTable
        title="Granger Causality Tests"
        rows={[
          {
            label: "Mobile subs -> Digital transactions",
            value: "F = 4.23, p = 0.018",
            significant: true,
          },
          {
            label: "Digital transactions -> Mobile subs",
            value: "F = 1.87, p = 0.162",
          },
          { label: "Optimal lag (AIC)", value: "3 months" },
          { label: "VAR stability", value: "All roots inside unit circle" },
          {
            label: "Johansen cointegration",
            value: "1 cointegrating vector (trace p = 0.003)",
            significant: true,
          },
        ]}
      />

      <div className="callout callout-green">
        <strong>Result:</strong> Mobile subscriber growth Granger-causes digital
        transaction growth (p = 0.018), but not vice versa. This is consistent
        with the causal logic: people first acquire phones, then adopt digital
        payments. The reverse channel (digital payment convenience driving phone
        adoption) is not statistically supported. The systems are also
        cointegrated, implying a long-run equilibrium relationship.
      </div>

      <Figure
        src="/figures/obj3_granger_causality.png"
        alt="Granger causality analysis visualization"
        caption="Figure 6.3: Impulse response functions from the bivariate VAR showing the asymmetric causal relationship between connectivity and digital payments."
      />

      <h2>The COVID Accelerator</h2>

      <p>
        A notable feature of the digital payments data is the COVID-19 effect.
        While total transactions dipped briefly during the strict April 2020
        lockdown (by approximately 27%), the recovery was V-shaped and
        accelerated the pre-existing trend. By July 2020&mdash;just three months
        after the lockdown&mdash;volumes had recovered to pre-COVID levels.
        By December 2020, they exceeded pre-COVID projections.
      </p>

      <p>
        COVID appears to have permanently shifted consumer habits: the share
        of cash transactions in the economy fell substantially and did not
        recover. This &quot;ratchet effect&quot; suggests that once consumers
        overcome the initial adoption barrier (forced by lockdown conditions),
        they do not revert to cash even when physical commerce resumes.
      </p>

      <h2>Key Takeaways</h2>

      <ol>
        <li>
          UPI grew from negligible to dominant in five years, a pace of
          infrastructure adoption unprecedented in financial history.
        </li>
        <li>
          Mobile connectivity Granger-causes digital payment adoption (p =
          0.018), establishing a temporal ordering consistent with the causal
          mechanism.
        </li>
        <li>
          The Jio+UPI combination created a flywheel: cheap data enabled
          payment apps, which increased perceived value of smartphones, which
          drove further adoption.
        </li>
        <li>
          COVID accelerated the structural shift by forcing first-time digital
          adoption among reluctant users.
        </li>
      </ol>

      <div className="callout callout-blue">
        <strong>Policy Implication:</strong> India&apos;s digital payments
        success demonstrates that public digital infrastructure (UPI) combined
        with private market competition (Jio driving cheap data) can achieve
        financial inclusion at scale. The model is being studied and replicated
        by dozens of countries globally.
      </div>
    </article>
  );
}
