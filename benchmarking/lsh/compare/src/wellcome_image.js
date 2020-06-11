import React, { Component } from "react";

class WellcomeImage extends Component {
  render() {
    let image;
    if (this.props.id) {
      image = (
        <img
          alt=""
          src={`https://iiif.wellcomecollection.org/image/${this.props.id}.jpg/full/400,400/0/default.jpg`}
        ></img>
      );
    }
    return <div>{image}</div>;
  }
}

export default WellcomeImage;
