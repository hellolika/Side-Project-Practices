TRUNCATE TABLE [dbo].[LocationDetail]
GO

BEGIN
	INSERT INTO [dbo].[LocationDetail] (
	[LocationName],
	[Latitude],
	[Longitude],
	[Range]
	) VALUES
('TechBodia', '11.5337918', '104.9772619', 500)

END