void setup()
{
  size(1000, 1000);
  background(0);
  noLoop();
}


color random_color()
{
  color c = color(random(0, 255), random(0, 255), random(0, 255));
  return c;
}

void draw()
{
  for(int i=0;i<10;i++)
  {
    color c = random_color();
    fill(c);
    circle(500, 500, 1000-100*i);
  }
  save("circle.png");
}
