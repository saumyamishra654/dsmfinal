import StatTable from "@/components/StatTable";

export default function EquityPage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">The Equity Question</h1>
        <p>
          Does telecommunications expansion improve educational inclusion for
          SC/ST communities?
        </p>
      </div>

      <h2>Constructing an Inclusion Metric</h2>

      <p>
        India&apos;s Scheduled Castes (SC) and Scheduled Tribes (ST) have
        historically been underrepresented in higher education. We construct an
        &quot;inclusion ratio&quot; defined as the share of SC/ST students in
        higher education relative to their share of the state population. A
        ratio of 1.0 indicates perfect proportional representation; values below
        1.0 indicate underrepresentation.
      </p>

      <p>
        The hypothesis is intuitive: mobile connectivity reduces information
        asymmetries that disproportionately affect marginalized communities.
        SC/ST students in remote areas may lack awareness of scholarship
        programmes, admission procedures, or available institutions. Cheap
        mobile internet could bridge this information gap, improving
        representation over time.
      </p>

      <p>
        We regress changes in the inclusion ratio on changes in teledensity,
        controlling for state fixed effects, year effects, and baseline economic
        indicators. The specification is demanding: we ask whether states that
        experienced larger increases in connectivity also saw larger
        improvements in SC/ST inclusion.
      </p>

      <h2>Results: Suggestive but Not Robust</h2>

      <StatTable
        title="SC/ST Inclusion Ratio and Teledensity"
        rows={[
          { label: "Coefficient (teledensity on inclusion)", value: "+0.0023" },
          { label: "Asymptotic p-value", value: "0.057" },
          { label: "Bootstrap p-value (1000 reps)", value: "0.125" },
          { label: "Wild cluster bootstrap", value: "p = 0.143" },
          { label: "Permutation test (5000 perms)", value: "p = 0.089" },
          { label: "Effect size", value: "Small (d = 0.28)" },
        ]}
      />

      <div className="callout callout-amber">
        <strong>Assessment:</strong> The asymptotic p-value of 0.057 is
        tantalizingly close to conventional significance, but every robustness
        check pushes the estimate further from significance. The bootstrap
        p-value of 0.125 and wild cluster bootstrap of 0.143 suggest that the
        asymptotic approximation is mildly optimistic. We classify this result
        as <em>suggestive but not robust</em>.
      </div>

      <h2>Why the Honest Null Matters</h2>

      <p>
        In a publication-biased academic environment, there is strong temptation
        to present the p = 0.057 result as &quot;marginally significant&quot;
        and move on. We resist this for several reasons:
      </p>

      <ol>
        <li>
          <strong>The bootstrap disagrees:</strong> When the asymptotic and
          bootstrap p-values diverge substantially (0.057 vs. 0.125), the
          bootstrap is generally more trustworthy. It makes fewer distributional
          assumptions and is better calibrated in small samples.
        </li>
        <li>
          <strong>Multiple testing:</strong> Across all objectives in this
          report, we conduct dozens of hypothesis tests. Without formal
          multiple-testing correction, a few &quot;p &lt; 0.10&quot; results are
          expected by chance alone.
        </li>
        <li>
          <strong>The effect size is small:</strong> Even if the result were
          significant, the economic magnitude is modest. A 10-point increase in
          teledensity is associated with a 0.023 increase in the inclusion
          ratio&mdash;barely perceptible against a mean ratio of approximately
          0.75.
        </li>
      </ol>

      <h2>Decomposing the Non-Result</h2>

      <p>
        Several mechanisms could explain why connectivity gains have not
        (yet) translated into measurable equity improvements:
      </p>

      <p>
        <strong>Structural barriers dominate information barriers:</strong> SC/ST
        underrepresentation in higher education may be driven more by economic
        constraints (inability to forgo earnings during study), geographic
        isolation from institutions, or social discrimination in admissions
        than by lack of information. In that case, connectivity addresses a
        secondary barrier while the primary barriers remain binding.
      </p>

      <p>
        <strong>Time lags:</strong> Educational decisions (enrolment in higher
        education) reflect decisions made years earlier (completing secondary
        school, passing entrance exams). The effect of connectivity on SC/ST
        inclusion may operate with a lag longer than our 2013-2022 window can
        capture. A student who gains internet access at age 14 might not
        appear in GER data until age 18-20.
      </p>

      <p>
        <strong>Content accessibility:</strong> Having a smartphone does not
        guarantee access to educationally relevant content in local languages,
        or digital literacy sufficient to navigate complex application
        processes. The &quot;last mile&quot; of digital inclusion extends beyond
        mere connectivity.
      </p>

      <h2>Subgroup Heterogeneity</h2>

      <StatTable
        title="Heterogeneity Analysis"
        rows={[
          { label: "SC inclusion (separately)", value: "beta = +0.0031, p = 0.082" },
          { label: "ST inclusion (separately)", value: "beta = +0.0009, p = 0.412" },
          { label: "Urban states subset", value: "beta = +0.0041, p = 0.103" },
          { label: "Rural states subset", value: "beta = +0.0012, p = 0.337" },
          { label: "Post-2016 only", value: "beta = +0.0028, p = 0.091" },
        ]}
      />

      <p>
        Interesting patterns emerge in subgroup analysis: the effect is
        somewhat stronger for SC communities than ST communities, and somewhat
        stronger in more urbanized states. This is consistent with ST communities
        facing additional barriers (geographic remoteness, language differences)
        that connectivity alone cannot address. However, none of these subgroup
        analyses achieve robust significance after accounting for the
        additional multiple testing.
      </p>

      <h2>Comparison with Literature</h2>

      <p>
        Our null result is not anomalous in the literature on ICT and equity.
        Aker &amp; Mbiti (2010) find positive effects of mobile phones on
        agricultural outcomes in Sub-Saharan Africa but null effects on
        educational metrics. Jensen (2007) documents that information provision
        via phones changes economic behavior but primarily in market
        contexts where information asymmetry is the binding constraint.
        Educational participation appears to be constrained by multiple
        simultaneous barriers, of which information is only one.
      </p>

      <div className="callout callout-blue">
        <strong>Conclusion:</strong> We find weak, non-robust evidence that
        teledensity expansion is associated with improved SC/ST educational
        inclusion. The direction is consistently positive across specifications,
        but the magnitude is small and statistical significance is not achieved
        under rigorous inference. This result should motivate further
        investigation with longer time series and quasi-experimental designs,
        not policy conclusions.
      </div>
    </article>
  );
}
