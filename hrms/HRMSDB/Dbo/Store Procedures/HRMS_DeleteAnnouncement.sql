CREATE PROCEDURE [dbo].[HRMS_DeleteAnnouncement.1.0.0]
	@id AS INT
AS
BEGIN 
	SET NOCOUNT ON

	IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Announcement] WHERE [Id] = @id)
	BEGIN
		SELECT 31 AS ErrorCode;
	END
	ELSE
	BEGIN
		DELETE FROM [dbo].[Announcement] WHERE [Id] = @id;
		SELECT 0 AS ErrorCode;
	END
END