// @flow
import Link from "next/link";
type Props = { title: string };

const Header = ({ title }: Props) => {
  return (
    <div
      style={{
        background: "#ffce3c",
        marginTop: "30px",
        marginBottom: "30px",
        position: "relative",
        height: "60px",
        display: "flex"
      }}
    >
      <h1 style={{ margin: 0, padding: 0 }}>
        <a href="/">
          <img
            src="https://i.wellcomecollection.org/assets/icons/android-chrome-512x512.png"
            width={90}
            height={90}
            alt={"Wellcome Collection logo"}
            title={"Wellcome Collection"}
            style={{
              marginTop: "-15px",
              marginLeft: "18px"
            }}
          />
        </a>
      </h1>
      <div style={{ display: "flex" }}>
        <h2
          style={{
            margin: 0,
            padding: 0,
            marginLeft: "18px",
            color: "#323232",
            lineHeight: "2.5em",
            minWidth: "150px"
          }}
        >
          {title}
        </h2>

        <ul
          style={{
            display: "flex",
            listStyle: "none",
            margin: 0,
            padding: 0,
            alignItems: "center"
          }}
        >
          <li
            style={{
              marginRight: "18px"
            }}
          >
            <Link href="/palette">
              <a
                style={{
                  color: "#323232",
                  textDecoration: "none",
                  fontWeight: "bold"
                }}
              >
                Palette
              </a>
            </Link>
          </li>
          <li
            style={{
              marginRight: "18px"
            }}
          >
            <Link href="/devise">
              <a
                style={{
                  color: "#323232",
                  textDecoration: "none",
                  fontWeight: "bold"
                }}
              >
                DeViSE
              </a>
            </Link>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default Header;
