const imagePreview = document.getElementById("image-preview");
      function readURL(input) {
        if (input.files && input.files[0]) {
          var reader = new FileReader();
          reader.onload = function (e) {
            imagePreview.innerHTML =
              '<img src="' +
              e.target.result +
              '" alt="Selected Image" id="imageResult" class="img-fluid rounded shadow-sm mx-auto d-block" />';
          };
          reader.readAsDataURL(input.files[0]);
        }
      }

      $(function () {
        $("#upload").on("change", function () {
          readURL(input);
        });
      });

      var input = document.getElementById("upload");

      input.addEventListener("change", showFileName);
      function showFileName(event) {
        var input = event.srcElement;
      }

      $(document).ready(function () {
        $("form").submit(function (e) {
          if ($("#upload")[0].files.length == 0) {
            e.preventDefault();
            alert("No file selected. Please select a file before submitting.");
          }
        });
      });

      document
        .getElementById("cancel-upload")
        .addEventListener("click", function (e) {
          e.preventDefault();
          document.getElementById("upload").value = "";
          imagePreview.innerHTML =
            '<label for="upload" class="btn btn btn-outline-light m-0 rounded-pill px-4 mx-auto w-20 custom-button text-uppercase font-weight-bold text-center">Choose file</label>';
        });
