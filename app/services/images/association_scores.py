"""`image_product_associations.overall_score` for links that are decided rather than measured.

Search multiplies every image hit by this score, and the column defaults to 0.0, so a link
written without one makes its image invisible to every image channel.
"""

PAGE_LINK_SCORE = 0.8          # same as entity linking's page_proximity
PRINTED_LABEL_SCORE = 0.9      # the page prints the series name next to the image
SOURCED_MATCH_SCORE = 0.95     # the maker's own image, confirmed by a visual match
