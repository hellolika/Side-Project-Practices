CREATE PROCEDURE [dbo].[HRMS_AddLocation.1.0.0]
	@locationName NVARCHAR(50),
	@latitude DECIMAL,
	@longitude DECIMAL,
	@range DECIMAL
AS
BEGIN
	SET NOCOUNT ON;

	INSERT INTO [dbo].[LocationDetail] ([LocationName], [Latitude], [Longitude], [Range])
	VALUES (@locationName, @latitude, @longitude, @range);

	SELECT 0 AS ErrorCode;
END