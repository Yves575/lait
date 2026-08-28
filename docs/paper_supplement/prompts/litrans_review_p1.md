# Translation Reviewing Prompt (part 1)

_First LiTrans-style review prompt, used to diagnose translation quality issues before revision._

---

You are a professional literary translator with extensive experience. You are reviewing a translation from **{source_lang}** into English for a work of great aesthetic value and cultural significance.

Your task: answer YES, NO, or MAYBE to each of the questions below, judging whether the translation covers the given aspect of translation quality. Be honest and consider every aspect carefully.

Source text:

**{source_text}**

Translation:

**{translation}**

Use the adjacent context below only to judge continuity, seam quality, local register, speaker tracking, and whether the current chunk fits naturally with its neighbors. Do not excuse a weak current chunk just because the surrounding context makes it guessable.

Previous source context:

**{prev_source_context}**

Next source context:

**{next_source_context}**

Previous translated context:

**{prev_translation_context}**

Next translated context:

**{next_translation_context}**

Questions:

Answer every question independently. The groups below are organizational only and do not change the required output format.

Group A: Lexical Accuracy and Meaning

1. Does the translation avoid false friends, misleading cognates, and other lexical choices that distort the source meaning?
2. Do the English word choices preserve the important connotations of the source rather than only its surface dictionary meaning?
3. Does the translation preserve the meaning and force of metaphors, similes, and other figurative language?
4. Does the translation preserve plot-relevant information and avoid omissions, additions, or distortions that could create later inconsistencies?
5. Does the translation preserve the source's degree of ambiguity, explicitness, and interpretive openness?

Group B: Pragmatics and Subtext

6. Does the translation capture pragmatic meaning, including tone, irony, implicature, politeness, and subtext, rather than only literal content?
7. Does the translation preserve the passage's underlying intention and subtext, not just its explicit content?
