window.onload = function () {
  new fullpage('#fullpage', {
    //options here
    autoScrolling:true,
    licenseKey: 'asd',
    navigation: true,
    navigationPosition: 'right',
    anchors: ['intro', 'pilot', 'engineer', 'posts', 'connect'],
    menu: '#fp-menu',
    verticalCentered: false,
  });



  // grab an element
  var myElement = document.querySelector("header#header");
  // construct an instance of Headroom, passing the element
  var options = {
      offset: 500,
      tolerance: 10
  }
  headroom = new Headroom(myElement, options);
  // initialise
  headroom.init();
}
