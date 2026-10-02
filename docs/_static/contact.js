/* Dismiss behavior for the "Contact us" widget (docs/_templates/contact.html).
 *
 * The widget is a <details> element, which only closes when its own <summary>
 * is clicked. Close it on an outside click or the Escape key as well, so that
 * it behaves the way a popover is expected to.
 */
(function () {
  "use strict";

  function setup() {
    var details = document.querySelector(".nimare-contact-details");
    if (!details) {
      return;
    }

    document.addEventListener("click", function (event) {
      if (details.open && !details.contains(event.target)) {
        details.open = false;
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && details.open) {
        details.open = false;
        details.querySelector("summary").focus();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
