.. _decoding methods:

.. include:: links.rst

Meta-analytic functional decoding
=================================

Discrete decoding
-----------------

Discrete decoding approaches characterize subsets of the Dataset or regions of interest, rather 
than continuous maps.

The BrainMap approach
`````````````````````
:func:`nimare.decode.discrete.BrainMapDecoder`, :func:`nimare.decode.discrete.brainmap_decode`

The BrainMap method for discrete functional decoding performs both forward and reverse inference
using an annotated coordinate-based database and a target sample of studies within that database.
Unlike the Neurosynth approach, the BrainMap approach incorporates information about the number
of foci associated with each study in the database.

1. Select studies in the database according to some criterion (e.g., having at least one peak in 
   an ROI).

2. For each label, studies in the database can now be divided into four groups.

   - Label-positive and selected --> :math:`S_{s+l+}`
   - Label-negative and selected --> :math:`S_{s+l-}`
   - Label-positive and unselected --> :math:`S_{s-l+}`
   - Label-negative and unselected --> :math:`S_{s-l-}`

3. Additionally, the number of foci associated with each of these groups is extracted.

   - Number of foci from studies with label, :math:`F_{l+}`
   - Number of foci from studies without label, :math:`F_{l-}`
   - Total number of foci in the database, :math:`F_{db} = F_{l+} + F_{l-}`

4. Compute the number of times any label is used in the database, :math:`L_{db}`
   (e.g., if every experiment in the database uses two labels,
   then this number is :math:`2S_{db}`, where :math:`S_{db}`
   is the total number of experiments in the database).

5. Compute the probability of being selected, :math:`P(s^{+})`.

   - :math:`P(s^{+}) = S_{s+} / F_{db}`, where :math:`S_{s+} = S_{s+l+} + S_{s+l-}`

6. For each label, compute the probability of having the label, :math:`P(l^{+})`.

   - :math:`P(l^{+}) = S_{l+} / L_{db}`, where :math:`S_{l+} = S_{s+l+} + S_{s-l+}`

7. For each label, compute the probability of being selected given presence of the label, 
   :math:`P(s^{+}|l^{+})`.

   - Can be re-interpreted as the probability of activating the ROI given a mental state.
   - :math:`P(s^{+}|l^{+}) = S_{s+l+} / F_{l+}`

8. Convert :math:`P(s^{+}|l^{+})` into the forward inference likelihood, :math:`\mathcal{L}`.

   - :math:`\mathcal{L} = P(s^{+}|l^{+}) / P(s^{+})`

9. Compute the probability of the label given selection, :math:`P(l^{+}|s^{+})`.

   - Can be re-interpreted as probability of a mental state given activation of the ROI.
   - :math:`P(l^{+}|s^{+}) = \frac{P(s^{+}|l^{+})P(l^{+})}{P(s^{+})}`
   - This is the reverse inference posterior probability.

10. Perform a binomial test to determine if the rate at which studies are selected from the
    set of studies with the label is significantly different from the base probability of
    studies being selected across the whole database.

    - The number of successes is :math:`\mathcal{K} = S_{s+l+}`, the number of trials is \
      :math:`n = F_{l+}`, and the hypothesized probability of success is :math:`p = P(s^{+})`
    - If :math:`S_{s+l+} < 5`, override the p-value from this test with 1, essentially ignoring \
      this label in the analysis.
    - Convert p-value to unsigned z-value.

11. Perform a two-way chi-square test to determine if presence of the label and selection are 
    independent.

    - If :math:`S_{s+l+} < 5`, override the p-value from this test with 1, essentially ignoring 
      this label in the analysis.
    - Convert p-value to unsigned z-value.


The Neurosynth approach
```````````````````````
:func:`nimare.decode.discrete.NeurosynthDecoder`, :func:`nimare.decode.discrete.neurosynth_decode`

The Neurosynth method for discrete functional decoding performs both forward and reverse inference
using an annotated coordinate-based database and a target sample of studies within that database.
Unlike the BrainMap approach, the Neurosynth approach uses an *a priori* value as the prior 
probability of any given experiment including a given label.

1. Select studies in the database according to some criterion (e.g., having at least one peak in 
   an ROI).

2. For each label, studies in the database can now be divided into four groups:

   - Label-positive and selected --> :math:`S_{s+l+}`
   - Label-negative and selected --> :math:`S_{s+l-}`
   - Label-positive and unselected --> :math:`S_{s-l+}`
   - Label-negative and unselected --> :math:`S_{s-l-}`

3. Set a prior probability :math:`p` of a given mental state occurring in the real world.

   - Neurosynth uses ``0.5`` as the default.

4. Compute :math:`P(s^{+})`:

   - Probability of being selected, :math:`P(s^{+}) = S_{s+} / (S_{s+} + S_{s-})`, where 
     :math:`S_{s+} = S_{s+l+} + S_{s+l-}` and :math:`S_{s-} = S_{s-l+} + S_{s-l-}`

5. For each label, compute :math:`P(l^{+})`:

   - :math:`P(l^{+}) = S_{l+} / (S_{l+} + S_{l-}`, where :math:`S_{l+} = S_{s+l+} + S_{s-l+}` 
     and :math:`S_{l-} = S_{s+l-} + S_{s-l-}`

6. Compute :math:`P(s^{+}|l^{+})`:

   - :math:`P(s^{+}|l^{+}) = S_{s+l+} / S_{l+}`

7. Compute :math:`P(s^{+}|l^{-})`:

   - :math:`P(s^{+}|l^{-}) = S_{s+l-} / S_{l-}`
   - Only used to determine sign of reverse inference z-value.

8. Compute :math:`P(s^{+}|l^{+}, p)`, where  is the prior probability of a label:

   - This is the forward inference posterior probability. Probability of selection given label and 
     given prior probability of label, :math:`p`.
   - :math:`P(s^{+}|l^{+}, p) = pP(s^{+}|l^{+}) + (1 - p)P(s^{+}|l^{-})`

9. Compute :math:`P(l^{+}|s^{+}, p)`:

   - This is the reverse inference posterior probability. Probability of label given selection and 
     given the prior probability of label.
   - :math:`P(l^{+}|s^{+}, p) = pP(s^{+}|l^{+}) / P(s^{+}|l^{+}, p)`

10. Perform a one-way chi-square test to determine if the rate at which studies are selected for a
    given label is significantly different from the average rate at which studies are selected 
    across labels.

    - Convert p-value to signed z-value using whether the number of studies selected for the
      label is greater than or less than the mean number of studies selected across labels to
      determine the sign.

11. Perform a two-way chi-square test to determine if presence of the label and selection are 
    independent.

    - Convert p-value to signed z-value using :math:`P(s^{+}|l^{-})` to determine sign.


The GC-LDA approach
```````````````````
:func:`nimare.decode.discrete.gclda_decode_roi`


The GC-LDA approach sums :math:`P(topic|voxel)` weights within the region of interest to produce 
topic-wise weights.

Continuous decoding
-------------------


Encoding
--------