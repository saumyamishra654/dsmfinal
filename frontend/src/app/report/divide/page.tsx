import Figure from "@/components/Figure";
import StatTable from "@/components/StatTable";

export default function DividePage() {
  return (
    <article className="prose">
      <div className="gradient-header">
        <h1 className="!text-white !mt-0 !mb-2">Who&apos;s Left Behind</h1>
        <p>
          PCA-based clustering, Louvain community detection, and the persistent
          digital divide across Indian states
        </p>
      </div>

      <h2>Multidimensional Digital Development</h2>

      <p>
        A single metric&mdash;teledensity or internet penetration&mdash;cannot
        fully capture the &quot;digital divide.&quot; States differ along multiple
        dimensions: mobile connectivity, internet penetration, digital payment
        adoption, educational outcomes, and urban-rural gaps. We use Principal
        Component Analysis (PCA) to reduce this high-dimensional space into
        interpretable composite indices, then apply Louvain community detection
        to identify natural clusters of states with similar digital profiles.
      </p>

      <p>
        The first two principal components capture approximately 72% of total
        variance. PC1 loads heavily on teledensity, internet penetration, and
        digital payments&mdash;a general &quot;digital development&quot; axis.
        PC2 loads on the gap between urban and rural connectivity and on
        educational variables&mdash;a &quot;digital equity&quot; axis. Together,
        they provide a meaningful two-dimensional map of India&apos;s digital
        landscape.
      </p>

      <Figure
        src="/figures/obj4_pca_biplot.png"
        alt="PCA biplot of state digital development"
        caption="Figure 7.1: PCA biplot showing states positioned by digital development (PC1) and digital equity (PC2). Variable loadings shown as vectors."
      />

      <h2>Community Detection: Natural Clusters</h2>

      <p>
        Rather than imposing arbitrary k-means clusters, we construct a
        k-nearest-neighbor graph of states based on their PCA scores and apply
        the Louvain algorithm for community detection. This identifies natural
        groupings that maximize within-cluster similarity and between-cluster
        difference, without requiring the researcher to pre-specify the number
        of groups.
      </p>

      <Figure
        src="/figures/obj4_louvain_graph.png"
        alt="Louvain community detection network graph"
        caption="Figure 7.2: Network graph with Louvain-detected communities. Edge thickness indicates similarity between states; colors indicate detected communities."
      />

      <p>
        The algorithm identifies three to four stable communities:
      </p>

      <ol>
        <li>
          <strong>Digital Leaders (Cluster 1):</strong> Delhi, Kerala, Tamil
          Nadu, Punjab, Karnataka, Himachal Pradesh. High teledensity, high
          internet penetration, high UPI adoption, high GER.
        </li>
        <li>
          <strong>Middle Tier (Cluster 2):</strong> Maharashtra, Gujarat, Andhra
          Pradesh, Rajasthan, Haryana, West Bengal. Moderate teledensity, growing
          digital payments, average educational outcomes.
        </li>
        <li>
          <strong>Lagging States (Cluster 3):</strong> Bihar, Jharkhand, Assam,
          Uttar Pradesh, Madhya Pradesh, Odisha. Low teledensity, low internet
          penetration, low digital payment adoption, low GER.
        </li>
      </ol>

      <Figure
        src="/figures/obj4_cluster_profiles.png"
        alt="Cluster profiles radar chart"
        caption="Figure 7.3: Radar chart showing mean values of key indicators for each detected cluster. The gap between leaders and laggards is substantial across all dimensions."
      />

      <h2>The Bihar Gap: 135 Years of Catch-Up</h2>

      <p>
        Perhaps the most striking finding from our convergence analysis, when
        combined with clustering, is the implied &quot;catch-up time&quot; for
        the most lagging states. Bihar, with a teledensity of approximately 55
        (compared to Delhi&apos;s 274), and a post-Jio growth rate that is not
        significantly different from Delhi&apos;s, would require approximately
        <strong>135 years</strong> to reach Delhi&apos;s current connectivity
        level, assuming current trends continue.
      </p>

      <div className="callout callout-red">
        <strong>The 135-Year Gap:</strong> At current post-Jio growth rates
        (where convergence has ceased), Bihar would need approximately 135 years
        to reach Delhi&apos;s present teledensity level. This is not a
        projection&mdash;it is a measure of how thoroughly the convergence
        mechanism has broken down. Without active intervention, these gaps are
        effectively permanent on any policy-relevant timescale.
      </div>

      <StatTable
        title="Digital Divide: Gap Analysis (2022)"
        rows={[
          { label: "Delhi teledensity", value: "273.7" },
          { label: "Bihar teledensity", value: "55.6" },
          { label: "Gap ratio", value: "4.9x", significant: true },
          { label: "Bihar CAGR (post-Jio)", value: "~1.5% per year" },
          { label: "Years to converge (linear)", value: "~135 years", significant: true },
          { label: "Leader cluster mean GER", value: "42.3" },
          { label: "Lagging cluster mean GER", value: "18.7" },
          { label: "GER gap ratio", value: "2.3x", significant: true },
        ]}
      />

      <h2>Persistence of Clustering</h2>

      <p>
        We compute the cluster assignments for each year from 2013 to 2022 and
        examine stability. The key finding: cluster membership is highly
        persistent. No state moves from the lagging cluster to the leader
        cluster during our observation period. Only two states
        (Odisha and West Bengal) move between middle and lagging tiers, and
        these transitions are marginal and temporary.
      </p>

      <p>
        This persistence connects directly to our convergence finding. The
        pre-Jio convergence was narrowing the gap between clusters, but the
        post-Jio stasis has locked states into their 2016 relative positions.
        The promise of mobile technology as a &quot;great equalizer&quot; has
        not materialized&mdash;at least not on a timeline relevant to current
        policy planning.
      </p>

      <h2>Connecting to the Convergence Story</h2>

      <p>
        The clustering results provide micro-foundations for the macro-level
        convergence finding. Convergence ceased not because all states reached
        similar levels, but because the growth acceleration from Jio was
        approximately uniform across clusters. States in all three clusters
        experienced a surge in 2016-2017 followed by similar plateau patterns.
        The absolute gaps remained intact even as all states grew.
      </p>

      <p>
        This &quot;parallel growth&quot; pattern is visible in the teledensity
        panel chart (Chapter 4): lines move up together but do not converge.
        Bihar&apos;s line remains far below Kerala&apos;s throughout the
        entire post-Jio period, growing at a similar rate but from a much
        lower base.
      </p>

      <h2>Policy Implications</h2>

      <ol>
        <li>
          <strong>Targeted infrastructure investment:</strong> The BharatNet
          programme (connecting gram panchayats with fiber) should prioritize
          Cluster 3 states, which face infrastructure rather than demand
          constraints.
        </li>
        <li>
          <strong>Beyond connectivity:</strong> For lagging states, simply
          adding cell towers is insufficient. Complementary investments in
          digital literacy, local-language content, and affordable devices are
          needed to convert connectivity into usage.
        </li>
        <li>
          <strong>Monitoring metrics:</strong> Policy should track not just
          subscriber counts (which can be inflated by multiple SIMs) but active
          data usage, app adoption rates, and transaction volumes as indicators
          of meaningful digital inclusion.
        </li>
      </ol>

      <div className="callout callout-blue">
        <strong>The Bottom Line:</strong> India&apos;s digital divide is
        multidimensional, persistent, and no longer self-correcting. The states
        that were digitally lagging in 2013 are still lagging in 2022, with
        cluster membership almost perfectly stable. Market forces alone will not
        close these gaps within any relevant policy horizon.
      </div>
    </article>
  );
}
